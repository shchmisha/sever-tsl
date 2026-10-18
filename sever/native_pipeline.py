"""Read every usable leaf directly from source pixels, without an OCR/type gate."""
from __future__ import annotations

import asyncio
from collections import Counter
from functools import partial
from pathlib import Path
import time

from . import __version__
from .agents import BatchBudgetExceeded, FatalProviderError
from .pipeline import export_csv, save_json, source_hash, trace
from .schema import DocumentResult, Evidence, FIELDS_BY_TYPE, FieldValue
from .tracing import archive_restart_artifacts


NATIVE_PIPELINE_VERSION = "native-image-joint-leaf-visibility-v1"


_SATURATION_POLICY = {
    "version": "full-source-exact-pixel-redundancy-v2-value-signature",
    "minimum_agreeing_fresh_readings": 3,
    "maximum_channel_difference": 0,
    "candidate_discovery_limit": None,
    "unresolved_status_is_preserved": True,
    "limitations": "Only pixel-identical complete native images are skipped; every changed, larger, sharper, or uncertain image is read afresh. Handheld video speed improvement is not asserted.",
}


_NO_PROGRESS_POLICY = {
    "version": "no-progress-leaf-stop-v1",
    "consecutive_views_without_progress": 3,
    "progress": "a new document-type or field-value/blank/conflict proposal, or any change in the fused type or field statuses",
    "counted_views": "completed and failed reads; quality-rejected and redundant candidates are not reads",
    "after_stop": "final joint reviews still run; remaining candidate views stay unread and are not claimed",
}


def _progress_state(readings, document, pending_type_review):
    """What the leaf has learned so far; a read that changes nothing is no progress."""
    from .image_fusion import _field_comparison

    proposals = set()
    for reading in readings:
        proposals.add(("type", reading.document_type))
        for name, field in reading.fields.items():
            if field.status == "recognized" and field.raw_value:
                proposals.add((name, "recognized", _field_comparison(name, field.raw_value)))
            elif field.status in {"not_present", "conflict"}:
                proposals.add((name, field.status))
    statuses = tuple(sorted((name, field.status) for name, field in document.fields.items()))
    return document.document_type, frozenset(proposals), statuses, pending_type_review is not None


def _reading_signature(reading, document):
    """Compare model proposals by status and value, without interpreting roles.

    Values use the same presentation-equivalence key as fusion. The literal
    evidence quote is excluded: a re-worded quote of an unchanged unreadable
    field is not a new proposal and must not keep the frame search alive.
    """
    from .image_fusion import _field_comparison

    if (reading.errors or document.errors or reading.document_type == "unknown"
            or document.document_type not in {"unknown", reading.document_type}
            or any(field.status == "conflict" for field in document.fields.values())
            or not any(field.status == "unreadable" for field in reading.fields.values())):
        return None
    return (reading.document_type, tuple((name, field.status,
        _field_comparison(name, field.raw_value) if field.raw_value else None)
        for name, field in sorted(reading.fields.items())))


def _redundant_pixels(candidate, reference):
    """A small changed glyph must not disappear in a whole-image mean score.

    No registration, resizing, cropping, or image-answer reuse is performed.
    Different geometry and unreadable image files remain uncertain and require
    a fresh model read. Every channel at every native source pixel must be
    identical; no averaged image metric establishes local non-improvement.
    """
    import numpy as np
    from PIL import Image

    proof = {"reference_frame_id": reference["frame_id"],
             "method": "complete_native_pixels_max_channel_difference"}
    try:
        with Image.open(candidate["path"]) as image:
            current = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(reference["path"]) as image:
            previous = np.asarray(image.convert("RGB"), dtype=np.uint8)
        proof.update(candidate_pixel_size=[current.shape[1], current.shape[0]],
                     reference_pixel_size=[previous.shape[1], previous.shape[0]],
                     candidate_file_sha256=source_hash(Path(candidate["path"])),
                     reference_file_sha256=source_hash(Path(reference["path"])))
        if current.shape != previous.shape:
            proof["decision"] = "different_source_geometry"
            return False, proof
        maximum = int(np.max(np.abs(current.astype(np.int16)-previous.astype(np.int16))))
        proof["maximum_channel_difference"] = maximum
        if maximum > _SATURATION_POLICY["maximum_channel_difference"]:
            proof["decision"] = "changed_local_source_pixels"
            return False, proof
        proof["decision"] = "redundant_source_pixels"
        return True, proof
    except (OSError, ValueError, TypeError, KeyError) as exc:
        proof.update(decision="uncertain_source_pixels", error_type=type(exc).__name__)
        return False, proof


class _FragmentSaturation:
    """Skip only redundant candidates after repeated unchanged model proposals."""
    def __init__(self):
        self.signature = None
        self.support = []
        self.skips = []

    def reset(self):
        self.signature = None
        self.support = []

    def observe(self, frame, reading, document):
        signature = _reading_signature(reading, document)
        if signature is None:
            self.reset()
            return
        if signature != self.signature:
            self.reset()
            self.signature = signature
        self.support.append(frame)
        self.support = self.support[-_SATURATION_POLICY["minimum_agreeing_fresh_readings"]:]

    def redundant(self, frame):
        minimum = _SATURATION_POLICY["minimum_agreeing_fresh_readings"]
        if len(self.support) < minimum:
            return None
        comparisons = []
        for reference in self.support:
            redundant, proof = _redundant_pixels(frame, reference)
            comparisons.append(proof)
            if not redundant:
                return None
        event = {"frame_id": frame["frame_id"], "timestamp_s": frame["timestamp_s"],
                 "frame_path": frame["path"], "supporting_read_frame_ids":
                 [reference["frame_id"] for reference in self.support],
                 "reason": "unchanged_unresolved_readings_and_redundant_source_pixels",
                 "pixel_comparisons": comparisons,
                 "unresolved_fields": [name for name, status, _ in self.signature[1]
                                       if status == "unreadable"],
                 "new_model_evidence": False}
        self.skips.append(event)
        return event

    def metadata(self):
        return {"policy": dict(_SATURATION_POLICY), "skipped_candidates": self.skips}


def _stop_reason(document, readings, views):
    from .image_fusion import _grounded_type_frames

    if document.document_type != "unknown":
        statuses = [field.status for field in document.fields.values()]
        if all(status in {"recognized", "not_present"} for status in statuses):
            return "all_required_fields_confirmed"
        if all(status in {"recognized", "not_present", "conflict"} for status in statuses):
            return "disagreements_preserved"
    supported = {reading.document_type for reading in readings if _grounded_type_frames(reading)}
    if len(supported) > 1:
        return "disagreeing_document_readings"
    # The model decides whether the visible form is outside the required scope.
    # Different spellings of its transcription do not overrule that decision.
    if readings and not supported:
        outside_frames = set()
        for reading, view in zip(readings, views):
            if view.get("model_document_type") != "unknown" or reading.errors:
                continue
            if (view.get("form_scope") != "out_of_scope"
                    or not str(view.get("observed_form") or "").strip()
                    or not str(view.get("visible_text") or "").strip()):
                continue
            outside_frames.update(e.frame_id for e in reading.classification_evidence)
        if len(outside_frames) >= 2:
            return "out_of_scope_form_confirmed"
    return None


def _conflict_review_frames(page, views, *, previous_frame_ids=(), required_frame_groups=()):
    """Cover readable disagreements with source pixels, ranking each side by quality.

    Four contexts bound one joint model context, not the number of source frames
    read for a leaf. A final review must include a newly read source view.
    Required groups preserve actual disputed evidence; quality ranks alternative
    pixels within a group, but cannot remove a readable side of the dispute.
    """
    known = {frame["frame_id"]: frame for frame in page["frames"]}
    candidates = []
    seen_ids, seen_hashes, seen_times = set(), set(), set()
    for view in views:
        frame = known.get(view.get("frame_id"))
        proof = view.get("pixel_proof") or {}
        rotation = proof.get("rotation_clockwise_degrees")
        digest = proof.get("source_sha256")
        if (not frame or type(rotation) is not int or rotation not in (0, 90, 180, 270)
                or not isinstance(digest, str) or len(digest) != 64):
            continue
        stamp = frame.get("timestamp_s")
        if (stamp is None or frame["frame_id"] in seen_ids
                or digest in seen_hashes or stamp in seen_times):
            continue
        seen_ids.add(frame["frame_id"])
        seen_hashes.add(digest)
        seen_times.add(stamp)
        candidates.append({**frame, "source_sha256": digest,
                           "rotation_clockwise_degrees": rotation})
    candidates.sort(key=lambda frame: (-frame.get("quality", {}).get("relative_selection_score", 0),
                                      frame["timestamp_s"], frame["frame_id"]))
    previous = set(previous_frame_ids)
    if previous:
        fresh = next((frame for frame in candidates if frame["frame_id"] not in previous), None)
        if fresh is None:
            return []
        candidates = [fresh, *(frame for frame in candidates if frame is not fresh)]
    groups = [set(group) for group in required_frame_groups]
    if groups:
        if previous:
            groups.append({frame["frame_id"] for frame in candidates if frame["frame_id"] not in previous})
        failed_coverages = set()

        def choose(chosen):
            chosen_ids = {frame["frame_id"] for frame in chosen}
            uncovered = tuple(index for index, group in enumerate(groups) if not group & chosen_ids)
            remaining = [groups[index] for index in uncovered]
            if not remaining:
                return chosen
            state = (uncovered, len(chosen))
            if len(chosen) == 4 or state in failed_coverages:
                return None
            group = min(remaining, key=lambda ids: sum(frame["frame_id"] in ids for frame in candidates))
            for frame in candidates:
                if frame["frame_id"] in group:
                    selected = choose([*chosen, frame])
                    if selected is not None:
                        return selected
            failed_coverages.add(state)
            return None

        required = choose([])
        if required is not None:
            chosen_ids = {frame["frame_id"] for frame in required}
            candidates = [*required, *(frame for frame in candidates if frame["frame_id"] not in chosen_ids)]
        # If all groups cannot fit, the caller must keep any uncovered field
        # unresolved. Returning quality-ranked pixels still lets it review
        # other fields whose disputed evidence is fully represented.
    return candidates[:4]


def _type_conflict_review_frames(page, readings, views, *, previous_frame_ids=(), extra_type_frames=None):
    """Represent every grounded disputed type with actual source pixels.

    Type tags guide driver coverage only; returned frames contain no readings,
    tags, suggested type or previous model values. Quality breaks ties within
    each group, and a final review must introduce a fresh source image.
    """
    from .image_fusion import _grounded_type_frames

    known = {frame["frame_id"]: frame for frame in page["frames"]}
    extra_type_frames = extra_type_frames or {}
    groups = {}
    candidates = {}
    for reading, view in zip(readings, views):
        kinds = ({reading.document_type} if _grounded_type_frames(reading) else set())
        kinds.update(kind for kind, ids in extra_type_frames.items() if view.get("frame_id") in ids)
        if not kinds:
            continue
        frame = known.get(view.get("frame_id"))
        proof = view.get("pixel_proof") or {}
        rotation, digest = proof.get("rotation_clockwise_degrees"), proof.get("source_sha256")
        if (not frame or type(rotation) is not int or rotation not in (0, 90, 180, 270)
                or not isinstance(digest, str) or len(digest) != 64
                or frame.get("timestamp_s") is None):
            continue
        candidate = {**frame, "source_sha256": digest, "rotation_clockwise_degrees": rotation}
        candidates[frame["frame_id"]] = candidate
        for kind in kinds:
            groups.setdefault(kind, []).append(candidate)
    required = {reading.document_type for reading in readings if _grounded_type_frames(reading)} | set(extra_type_frames)
    if len(required) < 2:
        # Pending heading confirmation also needs fresh pixels; field-bearing
        # readings with an omitted heading panel must not trigger an endless
        # full-field search without a targeted own-form review.
        return _conflict_review_frames(page, views, previous_frame_ids=previous_frame_ids)
    if set(groups) != required:
        return []
    rank = lambda frame: (-frame.get("quality", {}).get("relative_selection_score", 0),
                          frame["timestamp_s"], frame["frame_id"])
    for group in groups.values():
        group.sort(key=rank)
    previous = set(previous_frame_ids)

    def independent(frame, chosen):
        return all(frame["frame_id"] != other["frame_id"]
                   and frame["source_sha256"] != other["source_sha256"]
                   and frame["timestamp_s"] != other["timestamp_s"] for other in chosen)

    # At most four supported types. Bounded backtracking avoids a duplicate
    # SHA/PTS in an early group hiding an independent representative later.
    names = sorted(required)
    def choose_groups(index, chosen):
        if index == len(names):
            return chosen
        for frame in groups[names[index]]:
            if independent(frame, chosen):
                found = choose_groups(index + 1, [*chosen, frame])
                if found is not None:
                    return found
        return None

    chosen = choose_groups(0, [])
    if chosen is None:
        return []
    if previous and not any(frame["frame_id"] not in previous for frame in chosen):
        fresh = sorted((frame for frame in candidates.values() if frame["frame_id"] not in previous), key=rank)
        if len(chosen) < 4:
            new = next((frame for frame in fresh if independent(frame, chosen)), None)
            if new is not None:
                chosen.append(new)
        if not any(frame["frame_id"] not in previous for frame in chosen):
            # All four slots may be occupied by types. Replace that type's
            # representative, retaining every other disputed group.
            for new in fresh:
                group = next(name for name in names if any(frame["frame_id"] == new["frame_id"] for frame in groups[name]))
                index = names.index(group)
                others = [frame for position, frame in enumerate(chosen) if position != index]
                if independent(new, others):
                    chosen[index] = new
                    break
        if not any(frame["frame_id"] not in previous for frame in chosen):
            return []
    for frame in sorted(candidates.values(), key=rank):
        if len(chosen) == 4:
            break
        if independent(frame, chosen):
            chosen.append(frame)
    return sorted(chosen, key=rank)


async def process_video_images(source: Path, out: Path, *, config, sample_interval=.5,
                               max_pages=100, progress=None, reader=None,
                               additional_frames=None, conflict_reader=None,
                               no_progress_views=_NO_PROGRESS_POLICY["consecutive_views_without_progress"],
                               leaf_concurrency=1):
    from .video import analyze_video
    from .quality import evaluate_page_frames, assess_frame_quality
    from .frame_pool import iter_additional_page_frames
    from .image_fusion import fuse_readings, _grounded_type_frames
    from .image_agent import EvidencePersistenceError
    from .coverage import load_video_coverage
    from .report import write_report

    source, out = Path(source).resolve(), Path(out).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if type(leaf_concurrency) is not int or leaf_concurrency < 1:
        raise ValueError("leaf_concurrency must be a positive integer")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    before = config.usage.model_dump()
    source_id = source_hash(source)
    trace_path = out / "trace.jsonl"
    history = archive_restart_artifacts(out, ("partial-results.json", "results.json", "metrics.json",
        "fields.csv", "report.html", "selected-pages.json", "classifications.json"))
    trace(trace_path, "run_started", source=str(source), source_id=source_id, version=__version__,
          native_pipeline_version=NATIVE_PIPELINE_VERSION,
          sample_interval=sample_interval, document_reader="native_image", separate_classifier=False,
          local_ocr_used=False, previous_artifacts=str(history) if history else None)
    if progress:
        progress("Selecting stable page frames")
    pages = analyze_video(source, out / "video", sample_interval=sample_interval, max_pages=max_pages)
    video_coverage = load_video_coverage(out / "video" / "video-analysis.json")
    for page in pages:
        page["quality_gate"] = evaluate_page_frames(page)
        selected = page["quality_gate"].get("selected_frame_id")
        if selected:
            page["original_selected_frame_id"] = page["selected_frame_id"]
            page["selected_frame_id"] = selected
        trace(trace_path, "quality_gate", page_id=page["page_id"], decision=page["quality_gate"])
    save_json(out / "selected-pages.json", pages)
    # Leaves finish in any order when several run at once; exports keep page order.
    completed, in_flight, outcomes = {}, {}, {}
    default_reader = reader is None
    if reader is None:
        from .zoom_image_reader import read_frame_zoom
        reader = read_frame_zoom
        if conflict_reader is None:
            from .image_conflicts import read_leaf_conflicts
            conflict_reader = read_leaf_conflicts
    factory = additional_frames or iter_additional_page_frames

    def snapshot_pages():
        # An in-flight leaf appears once it has published its first result.
        return ([completed[index] for index in sorted(completed)]
                + [in_flight[index] for index in sorted(in_flight) if "result" in in_flight[index]])

    def consecutive_failures():
        """Failed leaves in a row over the finished prefix; quality-rejected leaves are neutral."""
        count = 0
        for index in range(1, len(pages) + 1):
            if index not in outcomes:
                break
            count = count + 1 if outcomes[index] == "failed" else 0 if outcomes[index] == "read" else count
            if count >= 3:
                break
        return count

    async def next_candidate(iterator):
        # Frame extraction runs ffmpeg; keep the event loop free for other leaves.
        if leaf_concurrency > 1:
            return await asyncio.to_thread(next, iterator, None)
        return next(iterator, None)

    async def process_page(position, page):
        page_id = page["page_id"]
        in_flight[position] = page
        # The default reader shares one upright rotation and one scope triage
        # across this leaf's views; custom readers keep their own signature.
        read = partial(reader, leaf_orientation={}, leaf_triage={}) if default_reader else reader
        trace(trace_path, "page_selected", **page)
        accepted = set(page["quality_gate"].get("accepted_frame_ids", []))
        ordered = sorted((f for f in page["frames"] if f["frame_id"] in accepted),
                         key=lambda f: (f["frame_id"] != page["selected_frame_id"],
                                        -f.get("quality", {}).get("relative_selection_score", 0)))
        readings, views, attempts = [], [], []
        document = DocumentResult(source_id=source_id, page_id=page_id, document_type="unknown")
        fusion = {}
        pool = None
        stop = None
        seen = set()
        saturation = _FragmentSaturation()
        conflict_reviews = []
        reviewed_type = None
        pending_type_review = None
        reviewed_fields = {}
        disputed_field_reviews = {}
        leaf_visibility_assessments = {}
        active_readings = []
        leaf_progress = {"state": None, "views_without_progress": 0, "triggered": False}

        def record_progress(read_completed):
            """Count consecutive reads that taught the leaf nothing new."""
            state = (_progress_state(readings, document, pending_type_review)
                     if read_completed else leaf_progress["state"])
            if read_completed and state != leaf_progress["state"]:
                leaf_progress.update(state=state, views_without_progress=0)
            else:
                leaf_progress["views_without_progress"] += 1
            if (no_progress_views is not None
                    and leaf_progress["views_without_progress"] >= no_progress_views):
                leaf_progress["triggered"] = True
            return leaf_progress["triggered"]

        def rebuild_fusion():
            nonlocal document, fusion, active_readings
            original_document, original_fusion = fuse_readings(readings, source_id=source_id, page_id=page_id)
            active_readings = readings
            if pending_type_review is not None:
                # A genuine joint pixel reading is evidence too. Later raw
                # proposals cannot silently overrule its contradictory form
                # after invalidation; only another fresh review can resolve it.
                evidence = [*original_document.classification_evidence,
                            *pending_type_review["classification_evidence"]]
                document = DocumentResult(source_id=source_id, page_id=page_id, document_type="unknown",
                    classification_evidence=evidence, notes=original_document.notes,
                    errors=list(dict.fromkeys([*original_document.errors, "image_fusion:conflicting_document_types"])))
                fusion = original_fusion
                fusion["pending_model_type_conflict"] = {
                    "document_type": pending_type_review["document_type"],
                    "frame_ids": sorted({item.frame_id for item in pending_type_review["classification_evidence"]}),
                    "resolution": "requires_fresh_joint_pixel_review"}
                fusion["type_confirmation"].update(status="conflict", confirmed=False, candidate_type=None)
                fusion["field_confirmations"] = {}
                fusion["remaining_fields"] = []
                fusion["coverage"].update(confirmed_fields=0, required_fields=0)
                return
            if reviewed_type is None:
                document, fusion = original_document, original_fusion
                apply_field_reviews()
                return
            kind = reviewed_type["document_type"]
            type_only = DocumentResult(source_id=source_id, page_id=page_id, document_type=kind,
                classification_evidence=reviewed_type["classification_evidence"],
                fields={name: FieldValue(status="unreadable", reason="This joint review read only the document type, not this field")
                        for name in FIELDS_BY_TYPE[kind]},
                notes=["Fresh joint source-image review adjudicated the type; its unreadable field placeholders are not field readings or independent model calls."])
            active_readings = [reading for reading in readings if reading.document_type == kind]
            active_readings = [*active_readings, type_only]
            document, fusion = fuse_readings(active_readings, source_id=source_id, page_id=page_id)
            # Every original answer and structural failure remains visible;
            # only a fresh model decision may exclude its disputed type tag
            # from current corroboration. No rejected-form field is mixed in.
            document.errors = list(dict.fromkeys([*document.errors,
                *(error for reading in readings for error in reading.errors)]))
            document.notes = list(dict.fromkeys([*document.notes,
                *(note for reading in readings for note in reading.notes)]))
            fusion["readings"] = original_fusion["readings"]
            fusion["coverage"].update(reading_count=original_fusion["coverage"]["reading_count"],
                distinct_frame_count=original_fusion["coverage"]["distinct_frame_count"],
                frame_ids=original_fusion["coverage"]["frame_ids"])
            fusion["type_confirmation"].update(resolution="fresh_joint_model_review",
                original_frame_ids_by_type=original_fusion["type_confirmation"]["frame_ids_by_type"],
                original_status=original_fusion["type_confirmation"]["status"],
                joint_review_frame_ids=sorted({e.frame_id for e in type_only.classification_evidence}),
                field_reads_in_type_review=False)
            fusion["model_type_adjudication"] = {"document_type": kind,
                "phase": reviewed_type["phase"], "source_reading_count": reviewed_type["source_reading_count"],
                "classification_evidence": [e.model_dump(mode="json") for e in type_only.classification_evidence],
                "prior_fusion_errors": original_document.errors,
                "fields_read": False, "method": "fresh_joint_source_images_not_majority_vote"}
            apply_field_reviews()

        def apply_field_reviews():
            """Keep fresh resolved fields until a later readable contradiction.

            Historical single-frame proposals remain intact in fusion.readings.
            A new unreadable observation cannot undo a completed pixel review.
            """
            from .image_fusion import _field_comparison

            for name, resolution in list(reviewed_fields.items()):
                if document.document_type != resolution["document_type"]:
                    continue
                field = resolution["field"]
                contradiction = None
                for reading in readings[resolution["source_reading_count"]:]:
                    if reading.document_type != document.document_type or name not in reading.fields:
                        continue
                    observed = reading.fields[name]
                    if not any(e.frame_id.strip() for e in observed.evidence):
                        continue
                    if (observed.status == "conflict"
                            or (observed.status in {"recognized", "not_present"} and observed.status != field.status)
                            or (observed.status == field.status == "recognized" and
                                _field_comparison(name, observed.raw_value) != _field_comparison(name, field.raw_value))):
                        contradiction = observed
                        break
                if contradiction is not None:
                    resolution["entry"].setdefault("invalidated_fields", {})[name] = {
                        "frame_ids": sorted({e.frame_id for e in contradiction.evidence}),
                        "reason": "New readable source evidence contradicts the prior joint field review"}
                    disputed_field_reviews[name] = resolution
                    del reviewed_fields[name]
                    continue
                document.fields[name] = field.model_copy(deep=True)
                fusion.setdefault("field_confirmations", {}).setdefault(name, {}).update(
                    status=field.status, confirmed=True, resolution="fresh_joint_model_review", reason=field.reason)
            for name, resolution in disputed_field_reviews.items():
                if document.document_type != resolution["document_type"] or name not in document.fields:
                    continue
                prior = resolution["field"]
                current = document.fields[name]
                if (current.status == "recognized"
                        and fusion.get("field_confirmations", {}).get(name, {}).get("strong_agreement")):
                    # Many single frames agree; an invalidated joint value does not
                    # keep the field in dispute. The review stays in the record.
                    fusion["field_confirmations"][name]["superseded_joint_review"] = prior.model_dump(mode="json")
                    continue
                reason = "A fresh joint pixel reading and later source reading disagree; another fresh joint review is required"
                document.fields[name] = FieldValue(status="conflict", reason=reason,
                    evidence=[*current.evidence, *prior.evidence])
                fusion.setdefault("field_confirmations", {}).setdefault(name, {}).update(
                    status="conflict", confirmed=False, resolution="requires_fresh_joint_pixel_review", reason=reason,
                    disputed_joint_reading=prior.model_dump(mode="json"))
            fusion.setdefault("coverage", {})["confirmed_fields"] = sum(
                field.status in {"recognized", "not_present"} for field in document.fields.values())
            fusion["remaining_fields"] = [name for name, field in document.fields.items()
                                           if field.status not in {"recognized", "not_present"}]

        def publish_snapshot(status="processing"):
            page["image_recognition"] = {"engine": "native_image", "views": views,
                "attempts": attempts, "complete": False, "fusion": fusion,
                "conflict_reviews": conflict_reviews,
                "leaf_visibility_assessments": leaf_visibility_assessments,
                "redundancy_saturation": saturation.metadata(),
                "local_ocr_used": False, "separate_classifier_used": False}
            page["result"] = document.model_dump(mode="json")
            snapshot = {"source_id": source_id, "status": status,
                "source": {"path": str(source), "sha256": source_id},
                "video_coverage": video_coverage,
                "pages": snapshot_pages(), "elapsed_seconds": round(time.monotonic()-started, 2)}
            save_json(out / "partial-results.json", snapshot)
            try:
                write_report(out, snapshot)
            except OSError as exc:
                trace(trace_path, "progress_report_failed", page_id=page_id,
                      error_type=type(exc).__name__)

        async def review_conflicts(phase):
            nonlocal document, fusion
            from .image_fusion import _field_comparison

            names = [name for name, field in document.fields.items() if field.status in {"conflict", "unreadable"}]
            if conflict_reader is None or not names or document.document_type == "unknown":
                return False
            required_groups = {}
            for name in names:
                groups = []
                resolution = disputed_field_reviews.get(name)
                if resolution is not None and resolution["document_type"] == document.document_type:
                    details = resolution["entry"]["answer"].get("adjudication", {}).get("field_reviews", {}).get(name, {})
                    support = details.get("independence", {}).get("supporting_frame_ids")
                    groups.append(set(support or [item.frame_id for item in resolution["field"].evidence]))
                if groups or document.fields[name].status == "conflict":
                    alternatives = {}
                    for reading in readings:
                        if reading.document_type != document.document_type or name not in reading.fields:
                            continue
                        field = reading.fields[name]
                        cited = {item.frame_id for item in field.evidence if item.frame_id.strip()}
                        if not cited:
                            continue
                        if field.status in {"recognized", "not_present"}:
                            key = (field.status, _field_comparison(name, field.raw_value)
                                   if field.status == "recognized" else None)
                            alternatives.setdefault(key, set()).update(cited)
                        elif field.status == "conflict":
                            groups.append(cited)
                    groups.extend(alternatives.values())
                required_groups[name] = groups
            previous = {identifier for entry in conflict_reviews if entry.get("kind", "fields") == "fields"
                        for identifier in entry["frame_ids"]}
            review_frames = _conflict_review_frames(page, views,
                previous_frame_ids=previous if phase == "final" else (),
                required_frame_groups=[group for groups in required_groups.values() for group in groups])
            if len(review_frames) < 2:
                return False
            frame_ids = {frame["frame_id"] for frame in review_frames}
            blocked_fields = {name: {"reason": "Joint review cannot cover every known readable side of this field's dispute",
                "uncovered_frame_groups": [sorted(group) for group in groups if not group & frame_ids]}
                for name, groups in required_groups.items() if any(not group & frame_ids for group in groups)}
            names = [name for name in names if name not in blocked_fields]
            entry = {"kind": "fields", "phase": phase, "status": "processing", "frame_ids": sorted(frame_ids),
                     "field_names": names, "frame_selection": "disputed_evidence_coverage_then_image_quality_and_time"}
            if blocked_fields:
                entry["blocked_fields"] = blocked_fields
            conflict_reviews.append(entry)
            if not names:
                entry["status"] = "skipped_incomplete_dispute_coverage"
                publish_snapshot()
                trace(trace_path, "image_conflict_review_skipped", page_id=page_id, **entry)
                return False
            publish_snapshot()
            trace(trace_path, "image_conflict_review_started", page_id=page_id, **entry)
            if progress:
                progress(f"Reviewing unresolved fields on image {position}/{len(pages)} using {len(review_frames)} source frames")
            try:
                # Only fresh source pixels and field names are passed. Original
                # single-frame proposals stay in the output, never model input.
                reviewed = await conflict_reader(review_frames, source_id=source_id, page_id=page_id,
                    document_type=document.document_type, unresolved_fields=names, config=config,
                    cache_dir=out / "image-reading-evidence", trace_path=trace_path)
                fields = {name: FieldValue.model_validate(value) for name, value in reviewed["fields"].items()}
                if set(fields) != set(names):
                    raise ValueError("Conflict review changed the requested field contract")
                for name, field in fields.items():
                    cited = {e.frame_id for e in field.evidence}
                    if not cited <= frame_ids:
                        raise ValueError("Conflict review cites a source outside the supplied frames")
                    if field.status in {"recognized", "not_present"} and len(cited) < 2:
                        raise ValueError("Conflict review needs two distinct actual source views")
                entry["answer"] = {**reviewed,
                    "fields": {name: field.model_dump(mode="json") for name, field in fields.items()}}
                for name, field in fields.items():
                    assessment = reviewed.get("adjudication", {}).get("field_reviews", {}).get(name, {}).get("additional_evidence", {})
                    cited = set(assessment.get("frame_ids", []))
                    prior_readable_evidence = (name in disputed_field_reviews or
                        document.fields[name].status != "unreadable" or any(
                            name in reading.fields and reading.fields[name].status in {"recognized", "not_present", "conflict"}
                            for reading in readings))
                    if (field.status == "unreadable" and assessment.get("terminal_for_current_leaf") is True
                            and not prior_readable_evidence
                            and assessment.get("action") == "obtain_other_document_side"
                            and len(cited) >= 2 and cited <= frame_ids
                            and cited <= {item.frame_id for item in field.evidence}):
                        leaf_visibility_assessments[name] = {**assessment, "phase": phase,
                            "source_reading_count": len(readings), "field_status": "unreadable",
                            "remaining_source_views_unread": phase != "final"}
                        document.fields[name] = field.model_copy(deep=True)
                    else:
                        leaf_visibility_assessments.pop(name, None)
                proposed_resolutions = [name for name, field in fields.items()
                                        if field.status in {"recognized", "not_present"}]
                # A joint review adjudicates between readings. When single frames
                # already agree on a value (two or more distinct frames), a joint
                # value that matches none of the single-frame readings is recorded
                # but not adopted; the field keeps its fused status. Where single
                # frames scatter or read nothing, the joint read may settle it.
                novel = {}
                for name, field in fields.items():
                    if field.status != "recognized" or not field.raw_value:
                        continue
                    frames_by_key = {}
                    for reading in readings:
                        if (reading.document_type != document.document_type or name not in reading.fields
                                or reading.fields[name].status != "recognized" or not reading.fields[name].raw_value):
                            continue
                        key = _field_comparison(name, reading.fields[name].raw_value)
                        frames_by_key.setdefault(key, set()).update(
                            e.frame_id for e in reading.fields[name].evidence if e.frame_id.strip())
                    dominant = any(len(frames) >= 2 for frames in frames_by_key.values())
                    if dominant and _field_comparison(name, field.raw_value) not in frames_by_key:
                        novel[name] = {"raw_value": field.raw_value, "single_frame_values": sorted(frames_by_key),
                                       "reason": "Joint-review value matches no single-frame reading although frames already agree; not adopted"}
                if novel:
                    entry["rejected_novel_values"] = novel
                replaced = []
                for name, field in fields.items():
                    if field.status not in {"recognized", "not_present"} or name in novel:
                        continue
                    reviewed_fields[name] = {"field": field.model_copy(deep=True),
                        "document_type": document.document_type, "source_reading_count": len(readings), "entry": entry}
                    disputed_field_reviews.pop(name, None)
                    document.fields[name] = field
                    replaced.append(name)
                    fusion.setdefault("field_confirmations", {}).setdefault(name, {}).update(
                        status=field.status, confirmed=True, resolution="fresh_joint_model_review",
                        reason=field.reason)
                fusion.setdefault("coverage", {})["confirmed_fields"] = sum(
                    field.status in {"recognized", "not_present"} for field in document.fields.values())
                fusion["remaining_fields"] = [name for name, field in document.fields.items()
                                               if field.status not in {"recognized", "not_present"}]
                entry.update(status="resolved" if len(replaced) == len(names) and not blocked_fields else "unresolved",
                             resolved_fields=replaced, proposed_resolved_fields=proposed_resolutions)
                if replaced:
                    document.notes.append("Unresolved fields were re-read jointly from actual source images; original single-frame proposals remain in conflict_reviews and views. This is model adjudication, not independent model agreement.")
                trace(trace_path, "image_conflict_review_finished", page_id=page_id,
                      phase=phase, status=entry["status"], resolved_fields=replaced)
                publish_snapshot()
                return bool(replaced)
            except (BatchBudgetExceeded, FatalProviderError) as exc:
                entry.update(status="failed", error_type=type(exc).__name__)
                document.errors.append("native_image: conflict_review_budget_or_provider_failure")
                try:
                    publish_snapshot("failed")
                except Exception:
                    # A disk/report failure cannot authorize more model calls
                    # after the provider or shared budget has stopped the run.
                    pass
                raise
            except Exception as exc:
                entry.update(status="failed", error_type=type(exc).__name__)
                if isinstance(exc, EvidencePersistenceError):
                    entry["evidence_persistence_failed"] = True
                document.errors.append(f"native_image: conflict_review_failed:{type(exc).__name__}")
                trace(trace_path, "image_conflict_review_failed", page_id=page_id,
                      phase=phase, error_type=type(exc).__name__)
                publish_snapshot()
                return False

        async def review_type_conflicts(phase):
            nonlocal reviewed_type, pending_type_review
            if conflict_reader is None or reviewed_type is not None:
                return False
            if (document.document_type != "unknown" or len(readings) < 2
                    or not any(reading.document_type != "unknown" for reading in readings)):
                return False
            previous = {identifier for entry in conflict_reviews if entry.get("kind") == "document_type"
                        for identifier in entry["frame_ids"]}
            review_frames = _type_conflict_review_frames(page, readings, views,
                previous_frame_ids=previous if phase == "final" else (),
                extra_type_frames=({pending_type_review["document_type"]:
                    {item.frame_id for item in pending_type_review["classification_evidence"]}}
                    if pending_type_review is not None else None))
            if len(review_frames) < 2:
                return False
            frame_ids = {frame["frame_id"] for frame in review_frames}
            entry = {"kind": "document_type", "phase": phase, "status": "processing",
                "frame_ids": sorted(frame_ids), "field_names": [],
                "frame_selection": "best_image_per_grounded_disputed_type_then_quality_and_time",
                "source_reading_count": len(readings)}
            conflict_reviews.append(entry)
            publish_snapshot()
            trace(trace_path, "image_type_conflict_review_started", page_id=page_id, **entry)
            if progress:
                progress(f"Reviewing the document type on image {position}/{len(pages)} using {len(review_frames)} source frames")
            try:
                reviewed = await conflict_reader(review_frames, source_id=source_id, page_id=page_id,
                    document_type="unknown", unresolved_fields=[], config=config,
                    cache_dir=out / "image-reading-evidence", trace_path=trace_path)
                if reviewed.get("fields") != {}:
                    raise ValueError("A type-only review must not supply field readings")
                kind = reviewed.get("document_type")
                if kind not in FIELDS_BY_TYPE:
                    raise ValueError("Type review returned an invalid document type")
                evidence = [Evidence.model_validate(item) for item in reviewed.get("classification_evidence", [])]
                cited = {item.frame_id for item in evidence}
                if not cited <= frame_ids:
                    raise ValueError("Type review cites a source outside the supplied frames")
                adjudication = reviewed.get("adjudication") or {}
                resolved = (kind != "unknown" and adjudication.get("status") == "resolved"
                            and adjudication.get("resolved_document_type") == kind)
                if resolved and len(cited) < 2:
                    raise ValueError("Type review needs two distinct actual source views")
                entry["answer"] = {**reviewed, "classification_evidence": [item.model_dump(mode="json") for item in evidence]}
                entry["status"] = "resolved" if resolved else "unresolved"
                if resolved:
                    reviewed_type = {"document_type": kind, "classification_evidence": evidence,
                        "phase": phase, "source_reading_count": len(readings), "entry": entry}
                    pending_type_review = None
                    rebuild_fusion()
                trace(trace_path, "image_type_conflict_review_finished", page_id=page_id,
                    phase=phase, status=entry["status"], document_type=kind)
                publish_snapshot()
                return resolved
            except (BatchBudgetExceeded, FatalProviderError) as exc:
                entry.update(status="failed", error_type=type(exc).__name__)
                document.errors.append("native_image: type_conflict_review_budget_or_provider_failure")
                try:
                    publish_snapshot("failed")
                except Exception:
                    pass
                raise
            except Exception as exc:
                entry.update(status="failed", error_type=type(exc).__name__)
                if isinstance(exc, EvidencePersistenceError):
                    entry["evidence_persistence_failed"] = True
                document.errors.append(f"native_image: type_conflict_review_failed:{type(exc).__name__}")
                trace(trace_path, "image_type_conflict_review_failed", page_id=page_id,
                    phase=phase, error_type=type(exc).__name__)
                publish_snapshot()
                return False

        def candidates():
            nonlocal pool
            yield from ordered
            if ordered:
                pool = iter(factory(source, page, out / "additional-frames"))
                yield from pool

        def requires_other_document_side():
            if (document.document_type == "unknown" or document.errors or pending_type_review is not None
                    or disputed_field_reviews or any(field.status == "conflict" for field in document.fields.values())):
                return False
            if any(attempt.get("status") in {"failed", "stopped"} for attempt in attempts):
                return False
            if any(entry.get("status") == "failed" for entry in conflict_reviews):
                return False
            unresolved = {name for name, field in document.fields.items()
                          if field.status not in {"recognized", "not_present"}}
            if not unresolved or any(document.fields[name].status != "unreadable" for name in unresolved):
                return False
            if any(name not in leaf_visibility_assessments for name in unresolved):
                return False
            if any(leaf_visibility_assessments[name]["source_reading_count"] != len(readings) for name in unresolved):
                return False
            # A prior readable/blank/conflicting proposal cannot be silenced by
            # a later "other side" explanation. Keep searching/reviewing it.
            return not any(reading.errors or any(field.status == "conflict" or
                (name in unresolved and field.status in {"recognized", "not_present"})
                for name, field in reading.fields.items()) for reading in readings)

        iterator = candidates()
        try:
            while True:
                frame = await next_candidate(iterator)
                if frame is None:
                    break
                frame_id = frame["frame_id"]
                if frame_id in seen:
                    continue
                seen.add(frame_id)
                attempt = {"frame_id": frame_id, "timestamp_s": frame["timestamp_s"]}
                attempts.append(attempt)
                try:
                    if frame_id not in accepted:
                        quality = assess_frame_quality(frame)
                        if not any(f["frame_id"] == frame_id for f in page["frames"]):
                            page["frames"].append(frame)
                        page["quality_gate"].setdefault("frame_dispositions", []).append(quality)
                        if not quality["accepted"]:
                            page["quality_gate"].setdefault("rejected_frames", []).append(quality)
                            attempt.update(status="skipped", reason="quality_rejected")
                            continue
                        accepted.add(frame_id)
                        page["quality_gate"]["accepted_frame_ids"].append(frame_id)
                    redundant = saturation.redundant(frame)
                    if redundant:
                        attempt.update(status="skipped", **{key: value for key, value in redundant.items()
                                                           if key not in {"frame_id", "timestamp_s"}})
                        continue
                    if progress:
                        progress(f"Reading image {position}/{len(pages)} at {frame['timestamp_s']:.1f}s, view {len(views)+1}")
                    answer = await read(frame, source_id=source_id, page_id=page_id, config=config,
                                         cache_dir=out / "image-reading-evidence", trace_path=trace_path)
                    reading = answer["document"]
                    if isinstance(reading, dict):
                        reading = DocumentResult.model_validate(reading)
                    if reading.source_id != source_id or reading.page_id != page_id:
                        raise ValueError("Image reading belongs to a different source or leaf")
                    if any(e.frame_id != frame_id for e in [*reading.classification_evidence,
                            *(e for field in reading.fields.values() for e in field.evidence)]):
                        raise ValueError("Image evidence belongs to a different source frame")
                    view = {**answer["recognition"], "frame_id": frame_id,
                            "frame_path": frame["path"], "timestamp_s": frame["timestamp_s"],
                            "document_type": reading.document_type, "document": reading.model_dump(mode="json"),
                            "panels": answer.get("panels", [])}
                    readings.append(reading)
                    views.append(view)
                    if _grounded_type_frames(reading):
                        for name, resolution in list(reviewed_fields.items()):
                            if reading.document_type != resolution["document_type"]:
                                resolution["entry"].setdefault("invalidated_fields", {})[name] = {
                                    "frame_ids": [frame_id],
                                    "reason": "New grounded form contradicts the reviewed document type"}
                                disputed_field_reviews[name] = resolution
                                del reviewed_fields[name]
                    if (reviewed_type is not None and _grounded_type_frames(reading)
                            and reading.document_type != reviewed_type["document_type"]):
                        reviewed_type["entry"]["invalidated_by"] = {
                            "frame_id": frame_id, "document_type": reading.document_type,
                            "source_reading_count": len(readings),
                            "reason": "New grounded source-frame reading contradicts the prior joint type decision"}
                        pending_type_review = reviewed_type
                        reviewed_type = None
                        for name, resolution in reviewed_fields.items():
                            resolution["entry"].setdefault("invalidated_fields", {})[name] = {
                                "frame_ids": [frame_id], "reason": "New grounded form contradicts the reviewed document type"}
                            disputed_field_reviews[name] = resolution
                        reviewed_fields.clear()
                    rebuild_fusion()
                    saturation.observe(frame, reading, document)
                    attempt.update(status="read", document_type=reading.document_type,
                                   observed_form=view.get("observed_form"))
                    stop = _stop_reason(document, active_readings, views)
                    publish_snapshot()
                    if (document.document_type == "unknown" and stop != "out_of_scope_form_confirmed"
                            and len(readings) >= 2 and conflict_reader is not None):
                        if not any(entry.get("kind") == "document_type" for entry in conflict_reviews):
                            await review_type_conflicts("initial")
                        stop = _stop_reason(document, active_readings, views)
                        if stop == "disagreeing_document_readings":
                            # The first joint review is bounded, not the leaf
                            # search. Continue all eligible source frames.
                            stop = None
                    if (document.document_type != "unknown" and fusion.get("remaining_fields")
                            and len(readings) >= 2 and conflict_reader is not None):
                        if not any(entry.get("kind", "fields") == "fields" for entry in conflict_reviews):
                            await review_conflicts("initial")
                        stop = _stop_reason(document, active_readings, views)
                        if stop == "disagreements_preserved":
                            # Unresolved conflicts do not cap the source-frame
                            # search. Revisit once at exhaustion with new pixels.
                            stop = None
                    if stop is None and requires_other_document_side():
                        stop = "requires_other_document_side"
                    if stop:
                        break
                    if record_progress(True):
                        trace(trace_path, "leaf_search_stopped_without_progress", page_id=page_id,
                              views_without_progress=leaf_progress["views_without_progress"],
                              read_views=len(views))
                        break
                except (BatchBudgetExceeded, FatalProviderError):
                    attempt.update(status="stopped", reason="model_budget_or_provider_failure")
                    document.errors.append("native_image: model_budget_or_provider_failure")
                    try:
                        publish_snapshot("failed")
                    except Exception:
                        pass
                    raise
                except Exception as exc:
                    attempt.update(status="failed", error_type=type(exc).__name__)
                    if isinstance(exc, EvidencePersistenceError):
                        attempt["evidence_persistence_failed"] = True
                    saturation.reset()
                    if record_progress(False):
                        trace(trace_path, "leaf_search_stopped_without_progress", page_id=page_id,
                              views_without_progress=leaf_progress["views_without_progress"],
                              read_views=len(views))
                        break
                finally:
                    if attempt.get("status") == "stopped":
                        try:
                            trace(trace_path, "image_view_attempt", page_id=page_id, **attempt)
                        except Exception:
                            pass
                    else:
                        trace(trace_path, "image_view_attempt", page_id=page_id, **attempt)
            if stop is None and conflict_reader is not None:
                await review_type_conflicts("final")
                await review_conflicts("final")
                if document.document_type != "unknown" and all(
                        field.status in {"recognized", "not_present"} for field in document.fields.values()):
                    stop = "all_required_fields_confirmed_after_joint_review"
                elif requires_other_document_side():
                    stop = "requires_other_document_side"
        except (BatchBudgetExceeded, FatalProviderError):
            raise
        except Exception as exc:
            document.errors.append(f"native_image: additional_frame_pool_failed:{type(exc).__name__}")
            stop = "additional_frame_pool_failed"
        finally:
            iterator.close()
            if pool is not None and hasattr(pool, "close"):
                pool.close()
        if stop == "requires_other_document_side":
            document.notes.append("Для непрочитанных полей модель запросила другую сторону документа; поля оставлены нечитаемыми.")
        # A successful reading must not hide failed reads of other source views.
        # Keep these operational failures separate from legitimate unknown forms.
        document.errors.extend(
            f"native_image: image_reading_failed:{attempt['frame_id']}:{attempt['error_type']}"
            for attempt in attempts if attempt.get("status") == "failed"
        )
        document.errors = list(dict.fromkeys([*document.errors, *(
            f"native_image: conflict_review_failed:{entry['error_type']}"
            for entry in conflict_reviews if entry.get("status") == "failed"
        )]))
        # Source-view/review failures remain in their immutable per-attempt
        # history. A later fully corroborated pixel reading can recover the
        # current result without marking that completed page as still failed.
        source_errors = {error for reading in readings for error in reading.errors}
        source_errors.update(f"native_image: image_reading_failed:{attempt['frame_id']}:{attempt['error_type']}"
                             for attempt in attempts if attempt.get("status") == "failed")
        source_errors.update(f"native_image: {prefix}conflict_review_failed:{entry['error_type']}"
                             for entry in conflict_reviews if entry.get("status") == "failed"
                             for prefix in ("", "type_"))
        # Correct later values cannot restore a lost raw response or its source
        # proof. Keep persistence failures active instead of declaring the
        # audit trail recovered through successful recognition on other views.
        persistence_errors = {
            f"native_image: image_reading_failed:{attempt['frame_id']}:{attempt['error_type']}"
            for attempt in attempts if attempt.get("evidence_persistence_failed")}
        persistence_errors.update(f"native_image: {prefix}conflict_review_failed:{entry['error_type']}"
            for entry in conflict_reviews if entry.get("evidence_persistence_failed")
            for prefix in ("", "type_"))
        fully_confirmed = document.document_type != "unknown" and all(
            field.status in {"recognized", "not_present"} for field in document.fields.values())
        error_history = []
        active_errors = []
        for error in document.errors:
            field_name = error.split(":", 1)[0]
            recovered = (error in source_errors and error not in persistence_errors and (fully_confirmed
                or (field_name in document.fields and document.fields[field_name].status in {"recognized", "not_present"})
                or (document.document_type != "unknown" and error in {
                    "Model document type has no supplied-panel heading evidence",
                    "Heading cites a panel that was not supplied for this frame"})))
            error_history.append({"error": error, "recovered": recovered})
            if not recovered:
                active_errors.append(error)
        document.errors = active_errors
        if not views:
            reason = "quality_gate_rejected" if not ordered else "image_reading_failed"
            document.errors.append("native_image: " + reason)
            outcomes[position] = "failed" if ordered else "neutral"
        else:
            outcomes[position] = "read"
        exhausted = ("source_views_unread_after_no_progress" if leaf_progress["triggered"]
                     else "source_views_exhausted_after_redundant_candidates" if saturation.skips
                     else "source_views_exhausted")
        page["image_recognition"] = {"engine": "native_image", "model": config.vision_model_name,
            "local_ocr_used": False, "separate_classifier_used": False, "views": views,
            "attempts": attempts, "stop_reason": stop or exhausted, "frame_count_limit": None,
            "conflict_reviews": conflict_reviews, "error_history": error_history,
            "leaf_visibility_assessments": leaf_visibility_assessments,
            "redundancy_saturation": saturation.metadata(),
            "no_progress_stop": {"policy": {**_NO_PROGRESS_POLICY,
                                            "consecutive_views_without_progress": no_progress_views},
                                 "triggered": leaf_progress["triggered"],
                                 "views_without_progress": leaf_progress["views_without_progress"]},
            "fusion": fusion, "evidence_coordinates": "complete supplied pixel panels, not measured word boxes"}
        page["neighbor_comparison"] = {"status": stop or exhausted, "attempts": attempts,
            "frame_count_limit": None, "remaining_fields": [name for name, f in document.fields.items()
                                                           if f.status not in {"recognized", "not_present"}]}
        page["classification"] = document.model_dump(mode="json")
        page["result"] = document.model_dump(mode="json")
        page["inn_verification"] = {"status": "not_performed", "roles": {},
            "reason": "Native image reading does not yet provide separately corroborated role-bound printed INNs; registry match is not asserted"}
        save_json(out / "image-readings" / (page_id + ".json"), page["image_recognition"])
        completed[position] = page
        in_flight.pop(position, None)
        failures = consecutive_failures()
        snapshot = {"source_id": source_id,
            "status": "failed" if failures >= 3 else "processing",
            "source": {"path": str(source), "sha256": source_id},
            "video_coverage": video_coverage,
            "pages": snapshot_pages(), "elapsed_seconds": round(time.monotonic()-started, 2)}
        save_json(out / "partial-results.json", snapshot)
        try:
            write_report(out, snapshot)
        except OSError as exc:
            trace(trace_path, "progress_report_failed", page_id=page_id,
                  error_type=type(exc).__name__)
        if failures >= 3:
            trace(trace_path, "batch_stopped", reason="three_consecutive_image_reading_failures")
            raise RuntimeError("Three consecutive image readings failed; source checkpoints retained")

    if leaf_concurrency == 1:
        for position, page in enumerate(pages, 1):
            await process_page(position, page)
    else:
        queue = iter(enumerate(pages, 1))

        async def worker():
            for position, page in queue:
                await process_page(position, page)

        # Several leaves share one model server and one batch budget. The first
        # budget, provider or failure-run stop cancels the other leaves in flight.
        trace(trace_path, "leaf_concurrency", leaves_at_once=leaf_concurrency)
        tasks = [asyncio.create_task(worker()) for _ in range(min(leaf_concurrency, len(pages)))]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            failed = next((task for task in tasks if task.done() and not task.cancelled()
                           and task.exception() is not None), None)
            if failed is not None:
                raise failed.exception()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    processed = [completed[index] for index in sorted(completed)]
    after = config.usage.model_dump()
    usage = {key: value-before.get(key, 0) if isinstance(value, (int, float)) and not isinstance(value, bool)
             else value for key, value in after.items()}
    result = {"schema_version": 1, "version": __version__, "source": {"path": str(source), "sha256": source_id},
        "video_coverage": video_coverage,
        "configuration": {"sample_interval": sample_interval, "max_pages": max_pages,
            "native_pipeline_version": NATIVE_PIPELINE_VERSION,
            "model": config.model_name, "vision_model": config.vision_model_name,
            "model_backend": config.provider_kind, "model_base_url": config.base_url,
            "document_reader": "native_image", "local_ocr_used": False, "separate_classifier_used": False,
            "field_engine": "direct source-image reading", "per_leaf_frame_limit": None,
            "field_conflict_review": "fresh_joint_source_images" if conflict_reader else "disabled_for_custom_reader",
            "document_type_conflict_review": "fresh_joint_source_images_covering_disputed_types" if conflict_reader else "disabled_for_custom_reader",
            "redundancy_saturation": dict(_SATURATION_POLICY),
            "no_progress_stop": {**_NO_PROGRESS_POLICY, "consecutive_views_without_progress": no_progress_views},
            "leaf_concurrency": leaf_concurrency,
            "verify_inn": False, "verify_inn_requested": config.verify_inn},
        "pages": processed, "usage": usage, "batch_usage": after,
        "elapsed_seconds": round(time.monotonic()-started, 2),
        "unit": "visible leaf; model-native transcription and field reading across actual source frames",
        "reading_summary": {"leaves_with_image_readings": sum(bool(p["image_recognition"]["views"]) for p in processed),
                            "image_views_read": sum(len(p["image_recognition"]["views"]) for p in processed),
                            "types": dict(Counter(p["result"]["document_type"] for p in processed))},
        "limitations": ["Model transcription is preserved; it is not independent ground truth.",
            "Evidence coordinates refer to supplied image panels, never inferred word locations.",
            "Unknown form still retains literal model transcription.",
            "Customer acceptance and measured accuracy require independent reviewed labels."]}
    save_json(out / "selected-pages.json", [{k: v for k, v in p.items() if k not in
        {"result", "classification", "inn_verification", "image_recognition"}} for p in processed])
    save_json(out / "classifications.json", {p["page_id"]: p["classification"] for p in processed})
    try:
        export_csv(out, processed)
        write_report(out, result)
        save_json(out / "results.json", result)
        trace(trace_path, "run_completed", page_count=len(processed), elapsed_seconds=result["elapsed_seconds"], usage=usage)
    except BaseException:
        archive_restart_artifacts(out, ("results.json", "metrics.json", "fields.csv", "report.html"), reason="failed-finalization")
        raise
    return result
