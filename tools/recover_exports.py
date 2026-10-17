"""Recover exports from machine checkpoints and a source-bound cache audit.

No OCR, model, network, manual labels or expected-value files are consulted.
This is export recovery, never a claim that another complete cold run occurred.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sever.pipeline import export_csv, save_json, source_hash
from sever.report import write_report
from sever.schema import DocumentResult
from sever.tracing import append_trace_row, archive_restart_artifacts


class RecoveryRefused(ValueError):
    """A missing or inconsistent proof prevents any export promotion."""


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def decision_sha256(value, convention):
    if convention == "UTF-8 json.dumps(ensure_ascii=False, sort_keys=True), default separators":
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    elif convention == "UTF-8 json.dumps(ensure_ascii=False, sort_keys=True, separators=(comma,colon))":
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    else:
        raise RecoveryRefused("Missing or unsupported raw decision SHA-256 convention")
    return hashlib.sha256(encoded).hexdigest()


def require(condition, message):
    if not condition:
        raise RecoveryRefused(message)


class CheckedFiles:
    def __init__(self):
        self.files = {}

    def check(self, reference):
        require(isinstance(reference, dict) and reference.get("path") and reference.get("sha256"),
                "Missing path/SHA-256 proof")
        path = Path(reference["path"]).resolve()
        require(path.is_file(), f"Proof file missing: {path}")
        previous = self.files.get(path)
        require(previous is None or previous == reference["sha256"], f"Conflicting file proofs: {path}")
        if previous is None:
            require(source_hash(path) == reference["sha256"], f"File SHA-256 mismatch: {path}")
            self.files[path] = reference["sha256"]
        return path

    def load(self, reference):
        path = self.check(reference)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise RecoveryRefused(f"Invalid JSON proof: {path}") from exc

    def recheck(self):
        for path, expected in self.files.items():
            require(path.is_file() and source_hash(path) == expected, f"Proof changed during recovery: {path}")


def identity(page):
    return {key: page[key] for key in ("page_id", "start_s", "end_s", "selected_frame_id")}


def trace_snapshot(path, proof, source_id):
    raw = path.read_bytes()
    require(not raw or raw.endswith(b"\n"), "Current trace has an incomplete final line")
    try:
        rows = [json.loads(line) for line in raw.splitlines()]
    except (ValueError, UnicodeError) as exc:
        raise RecoveryRefused("Current trace contains malformed JSON") from exc
    starts = [index for index, row in enumerate(rows) if row.get("event") == "run_started"]
    require(starts, "No current run boundary in trace")
    boundary = starts[-1]
    require(boundary + 1 == proof.get("trace_run_started_line"), "Trace run boundary disagrees with audit")
    require(rows[boundary].get("source_id") == source_id, "Trace source disagrees with recovered source")
    counts = Counter(row.get("event") for row in rows[boundary:])
    require(counts["model_request_started"] == counts["model_response_received"] == 0,
            "Replay trace contains provider requests/responses; zero-paid recovery proof is insufficient")
    require(counts["run_completed"] == counts["exports_recovered"] == 0, "Current run already has a completion event")
    for row in rows[boundary:]:
        delta = row.get("usage_delta", {})
        require(all(not delta.get(key, 0) for key in ("requests", "input_tokens", "output_tokens", "actual_cost_rub")),
                "Replay trace contains nonzero provider usage")
    require(proof.get("paid_requests_since_boundary") == proof.get("paid_responses_since_boundary") == 0,
            "Audit lacks zero-paid replay proof")
    require(rows[boundary].get("sample_interval") == proof["configuration"].get("sample_interval"),
            "Replay sampling differs from archived configuration")

    def timestamp(row):
        value = row.get("time")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
        return None

    begin = timestamp(rows[boundary])
    times = [timestamp(row) for row in rows[boundary:]]
    require(begin is not None and all(value is not None for value in times), "Unusable replay timestamps")
    return {"run_started_line": boundary + 1, "event_count": len(rows) - boundary,
            "event_counts": dict(counts), "replay_duration_seconds": round(max(times) - begin, 2),
            "provider_usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0, "actual_cost_rub": 0}}


def validate_page(page, source_id, audit_page, files):
    frame_map = {frame["frame_id"]: frame for frame in page["frames"]}
    require(len(frame_map) == len(page["frames"]) and page["selected_frame_id"] in frame_map,
            "Duplicate frames or missing selected frame")
    for frame in frame_map.values():
        require(Path(frame["path"]).is_file(), f"Source frame missing: {frame['frame_id']}")
    require(audit_page.get("unchanged") is True and audit_page.get("parent_page_sha256") == canonical_sha256(page),
            f"Page unchanged proof disagrees: {page['page_id']}")
    require(audit_page.get("selected_frame_id") == page["selected_frame_id"], "Selected-frame audit disagrees")
    require(audit_page.get("classification_object") == page["classification"]
            and audit_page.get("classification_sha256") == canonical_sha256(page["classification"]),
            "Classification audit disagrees")
    observations = files.load(audit_page["ocr_artifact"])
    require(isinstance(observations, list), "OCR artifact must contain source observations")
    ocr_map = {view["frame_id"]: view for view in observations}
    require(len(ocr_map) == len(observations), "Duplicate OCR observation frame IDs")
    for frame_id, view in ocr_map.items():
        require(frame_id in frame_map, "OCR observation is not a frame of this leaf")
        processing = view.get("processing", {})
        require(processing.get("source_sha256"), "OCR view lacks source image SHA-256")
        files.check({"path": frame_map[frame_id]["path"], "sha256": processing["source_sha256"]})
        require(Path(processing["ocr_image_path"]).is_file(), "OCR-oriented image missing")
    for key in ("classification", "result"):
        document = DocumentResult.model_validate(page[key])
        require(document.source_id == source_id and document.page_id == page["page_id"], "Document identity mismatch")
        groups = [document.classification_evidence, *(field.evidence for field in document.fields.values())]
        for group in groups:
            for evidence in group:
                require(evidence.frame_id in frame_map and evidence.frame_id in ocr_map,
                        "Evidence references a frame without audited source OCR")
                if evidence.bbox is not None:
                    box = evidence.bbox
                    require(len(box) == 4 and all(math.isfinite(value) for value in box)
                            and min(box) >= 0 and box[2] > 0 and box[3] > 0
                            and box[0] + box[2] <= 1.000001 and box[1] + box[3] <= 1.000001,
                            "Evidence has invalid OCR coordinates")
                if evidence.region_index is not None:
                    regions = {region["region_index"]: region for region in ocr_map[evidence.frame_id]["regions"]}
                    require(evidence.region_index in regions, "Evidence region absent from actual OCR observation")
                    region = regions[evidence.region_index]
                    require(evidence.text == region["text"], "Evidence text differs from its recorded OCR region")
                    if evidence.bbox is not None:
                        require(max(abs(a - b) for a, b in zip(evidence.bbox, region["bbox"])) <= .006,
                                "Evidence coordinates differ from actual OCR region")
        require(all(field.evidence for field in document.fields.values() if field.status == "recognized"),
                "Recognized field has no source evidence")
    return ocr_map


def prepare_recovery(audit_path, audit_sha256):
    files = CheckedFiles()
    audit = files.load({"path": str(audit_path), "sha256": audit_sha256})
    require(audit.get("passed") is True and audit.get("parents_unchanged") is True,
            "Cached-reader audit did not pass")
    require(audit.get("truth_used") is False and audit.get("network_allowed") is False,
            "Audit is not an independent local machine-cache check")
    require(not audit.get("failures") and not audit.get("changes") and audit.get("usage", {}).get("requests") == 0,
            "Audit contains changed readings, failures or provider usage")
    proof = audit["recovery_inputs"]
    parent = files.load(proof["parent_results"])
    checkpoint = files.load(proof["checkpoint"])
    inventory = files.load(proof["inventory"])
    classifications = files.load(proof["classifications"])
    source = files.check(proof["source"])
    source_id = proof["source"]["sha256"]
    require(parent["source"]["sha256"] == checkpoint.get("source_id") == source_id
            and Path(parent["source"]["path"]).resolve() == source, "Source SHA/path differs")
    require(proof.get("configuration") == parent["configuration"]
            and proof.get("configuration_sha256") == canonical_sha256(parent["configuration"]),
            "Configuration/model differs from archived run")
    require(all(proof.get(key) is True for key in ("inventory_identity_unchanged", "classifications_unchanged",
                                                  "checkpoint_identity_matches_parent_prefix")), "Audit lacks inventory identity proof")
    full, prefix = parent["pages"], checkpoint["pages"]
    require(0 < len(prefix) < len(full), "Checkpoint must be a nonempty proper prefix")
    require(len(full) == proof.get("parent_page_count") == len(inventory)
            and len(prefix) == proof.get("checkpoint_page_count"), "Page counts disagree")
    ids = [page["page_id"] for page in full]
    require(len(set(ids)) == len(ids) and [page["page_id"] for page in inventory] == ids
            and [page["page_id"] for page in prefix] == ids[:len(prefix)], "Page order/IDs disagree")
    require(proof.get("expected_suffix_page_ids") == ids[len(prefix):], "Suffix page order disagrees")
    require(set(classifications) == set(ids), "Classification inventory differs")
    for index, (old, selected) in enumerate(zip(full, inventory)):
        require(identity(old) == identity(selected), "Selected inventory identity differs")
        require(old["start_s"] < old["end_s"] and (not index or full[index - 1]["end_s"] <= old["start_s"]),
                "Leaf intervals overlap or are unordered")
        old_frames = {frame["frame_id"]: frame for frame in old["frames"]}
        require(all(frame == old_frames.get(frame["frame_id"]) for frame in selected["frames"]),
                "Baseline frame inventory differs from archived run")
        require(old["classification"] == classifications[old["page_id"]], "Archived classification differs from current classifier")
        if index < len(prefix):
            require(identity(prefix[index]) == identity(old) and prefix[index]["classification"] == classifications[old["page_id"]],
                    "Checkpoint leaf identity/classification differs")
    sources = audit.get("sources", [])
    merged = deepcopy(prefix + full[len(prefix):])
    audited = {}
    for scope, expected, reference in (
        ("latest_machine_partial_prefix", prefix, proof["checkpoint"]),
        ("preserved_complete_suffix", full[len(prefix):], proof["parent_results"]),
    ):
        matches = [entry for entry in sources if entry.get("source_id") == source_id and entry.get("scope") == scope]
        require(len(matches) == 1, f"Missing unique audited scope: {scope}")
        entry = matches[0]
        require(Path(entry["parent_results_path"]).resolve() == Path(reference["path"]).resolve()
                and entry["parent_results_sha256"] == reference["sha256"], "Audited parent/checkpoint reference differs")
        require(entry["page_ids"] == [page["page_id"] for page in expected]
                and [page["page_id"] for page in entry["pages"]] == entry["page_ids"], "Audited page sequence differs")
        for page, audit_page in zip(expected, entry["pages"]):
            validate_page(page, source_id, audit_page, files)
            audited[page["page_id"]] = audit_page
    reader_model = proof["cache_reader_model"]
    require(parent["configuration"]["field_engine"] == f"selected role/header pixels via {reader_model}", "Reader model proof differs")
    readings = {}
    for reading in audit.get("cache_readings", []):
        if reading.get("source_id") != source_id or reading.get("page_id") not in ids:
            continue
        cache = files.load({"path": reading["cache_path"], "sha256": reading["cache_file_sha256"]})
        require(cache.get("status") == "success" and cache.get("source_id") == source_id
                and cache.get("page_id") == reading["page_id"], "Cache reading identity/status differs")
        require(cache.get("model") == reading.get("model") == reader_model
                and cache.get("base_url") == reading.get("base_url") == proof["cache_reader_base_url"], "Cache model/endpoint differs")
        require(cache.get("input_sha256") == reading.get("input_sha256")
                and decision_sha256(cache.get("decision"), audit.get("decision_sha256_convention")) == reading.get("decision_sha256")
                and cache.get("crop", {}).get("upload_sha256") == reading.get("context_upload_sha256")
                and cache.get("crop", {}).get("frame_id") == reading.get("frame_id"), "Cache input/decision proof differs")
        page = next(page for page in merged if page["page_id"] == reading["page_id"])
        frame = next((frame for frame in page["frames"] if frame["frame_id"] == reading["frame_id"]), None)
        require(frame is not None, "Audited cache reading frame not in this leaf")
        files.check({"path": frame["path"], "sha256": reading["ocr_source_image_sha256"]})
        files.check({"path": reading["ocr_image_path"], "sha256": cache["crop"]["oriented_image_sha256"]})
        readings[(reading["page_id"], reading["frame_id"])] = reading
    for page_id, audit_page in audited.items():
        require(all((page_id, frame_id) in readings for frame_id in audit_page.get("read_frame_ids", [])),
                "Audited page lacks a verified cached reader input")
        if classifications[page_id]["document_type"] != "unknown":
            require(audit_page.get("read_frame_ids"), "Target page lacks a cached reader proof")
    snapshot = trace_snapshot(files.check(proof["current_trace"]), proof, source_id)
    result = deepcopy(parent)
    result["pages"] = merged
    result["elapsed_semantics"] = "Archived full-run duration preserved; this is audited export recovery, not a fresh cold run"
    result["recovery"] = {
        "reason": "Interrupted export/replay recovered from exact machine checkpoint and unchanged source-bound cached readings",
        "source_complete_proven": True, "manual_ground_truth_used": False, "network_used": False,
        "parent_results": proof["parent_results"], "checkpoint": proof["checkpoint"],
        "audit": {"path": str(Path(audit_path).resolve()), "sha256": audit_sha256},
        "inventory": proof["inventory"], "classifications": proof["classifications"],
        "replay_trace": {**proof["current_trace"], "byte_length": Path(proof["current_trace"]["path"]).stat().st_size},
        "prefix_page_count": len(prefix), "suffix_page_count": len(full) - len(prefix),
        "parent_elapsed_seconds": parent["elapsed_seconds"], "replay_duration_seconds": snapshot["replay_duration_seconds"],
        "replay_provider_usage": snapshot["provider_usage"], "replay_trace_snapshot": snapshot,
        "parent_usage_semantics": "Preserved incurred usage of archived batch; replay usage recorded separately",
        "audited_ocr_artifacts": {page_id: value["ocr_artifact"] for page_id, value in audited.items()},
        "prefix_pages_sha256": canonical_sha256(prefix), "suffix_pages_sha256": canonical_sha256(full[len(prefix):]),
    }
    relocations = []
    for page in result["pages"]:
        audited_artifact = audited[page["page_id"]]["ocr_artifact"]
        if Path(page["ocr_artifact"]).resolve() != Path(audited_artifact["path"]).resolve():
            # Fix only the exported source-artifact link after original machine
            # page digests and every evidence frame have been verified.
            relocations.append({"page_id": page["page_id"], "original_path": page["ocr_artifact"],
                                "new_path": audited_artifact["path"], "new_sha256": audited_artifact["sha256"],
                                "original_page_sha256": audited[page["page_id"]]["parent_page_sha256"]})
            page["ocr_artifact"] = audited_artifact["path"]
    result["recovery"]["ocr_artifact_relocations"] = relocations
    result["recovery"]["exported_pages_sha256"] = canonical_sha256(result["pages"])
    return result, files


def recover_exports(audit_path, audit_sha256, out, *, check_only=False):
    started = time.monotonic()
    result, files = prepare_recovery(audit_path, audit_sha256)
    out = Path(out).resolve()
    names = ("fields.csv", "report.html", "selected-pages.json", "classifications.json", "recovery-provenance.json", "results.json")
    require(not (out / "results.json").exists(), "Current results.json exists; recovery cannot overwrite a completed export")
    require(not (out / "recovery-provenance.json").exists(), "Recovery provenance already exists")
    checkpoint = Path(result["recovery"]["checkpoint"]["path"]).resolve()
    trace = Path(result["recovery"]["replay_trace"]["path"]).resolve()
    require(checkpoint.parent == trace.parent == out, "Output must be the original checkpoint/trace run directory")
    if check_only:
        files.recheck()
        return {"checks_passed": True, "page_count": len(result["pages"]), "exports_written": False,
                "prefix_page_count": result["recovery"]["prefix_page_count"], "suffix_page_count": result["recovery"]["suffix_page_count"]}
    out.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".recover-exports-", dir=out))
    promoted = []
    try:
        export_csv(stage, result["pages"])
        # Report links must refer to the final run folder, not the temporary stage.
        write_report(stage, result, report_root=out)
        report = (stage / "report.html").read_text(encoding="utf-8")
        for frame in (frame for page in result["pages"] for frame in page["frames"]):
            old = html.escape(os.path.relpath(frame["path"], stage), quote=True)
            new = html.escape(os.path.relpath(frame["path"], out), quote=True)
            report = report.replace(f'href="{old}"', f'href="{new}"').replace(f'src="{old}"', f'src="{new}"')
        report = report.replace(" с обработки.", " с исходного полного прогона (не время восстановления).", 1)
        result["recovery"]["recovery_elapsed_seconds"] = round(time.monotonic() - started, 2)
        result["recovery"]["recovery_provenance_path"] = str(out / "recovery-provenance.json")
        notice = '<p><strong>Восстановленный экспорт.</strong> ' + html.escape(
            f"Первый {result['recovery']['prefix_page_count']} листов взят из текущего машинного checkpoint; "
            f"остальные {result['recovery']['suffix_page_count']} — из проверенного архивного результата. "
            f"Время восстановления экспорта: {result['recovery']['recovery_elapsed_seconds']} с. "
            f"Новых запросов провайдеру при проверенном replay: 0. Свежий полный прогон не выполнялся.") + '</p>'
        report = report.replace("<h1>Север ТСЛ — результат обработки</h1>", "<h1>Север ТСЛ — результат обработки</h1>" + notice)
        (stage / "report.html").write_text(report, encoding="utf-8")
        save_json(stage / "selected-pages.json", [{key: value for key, value in page.items()
                  if key not in {"result", "classification", "inn_verification"}} for page in result["pages"]])
        save_json(stage / "classifications.json", {page["page_id"]: page["classification"] for page in result["pages"]})
        save_json(stage / "recovery-provenance.json", result["recovery"])
        save_json(stage / "results.json", result)
        files.recheck()
        # Keep all prior files/checkpoints: only replace generated exports after
        # every proof and staged export has succeeded. Final JSON is last.
        history = archive_restart_artifacts(out, tuple(name for name in names if name not in {"results.json", "recovery-provenance.json"}),
                                            reason="before-export-recovery")
        if history:
            for key, name in (("inventory", "selected-pages.json"), ("classifications", "classifications.json")):
                preserved = history / name
                if preserved.is_file():
                    result["recovery"][key] = {**result["recovery"][key],
                        "original_path": result["recovery"][key]["path"], "path": str(preserved)}
            result["recovery"]["replaced_exports_history"] = str(history)
            save_json(stage / "recovery-provenance.json", result["recovery"])
            save_json(stage / "results.json", result)
        for name in names:
            (stage / name).replace(out / name)
            promoted.append(name)
        append_trace_row(trace, {"time": time.time(), "event": "exports_recovered", "source_id": result["source"]["sha256"],
            "source_complete_proven": True, "page_count": len(result["pages"]), "results_sha256": source_hash(out / "results.json"),
            "recovery_provenance_path": str(out / "recovery-provenance.json"), "replay_provider_calls": 0,
            "trace_input_byte_length": result["recovery"]["replay_trace"]["byte_length"],
            "trace_input_sha256": result["recovery"]["replay_trace"]["sha256"]})
        return result
    except BaseException:
        if promoted:
            archive_restart_artifacts(out, tuple(promoted), reason="failed-export-recovery")
        raise
    finally:
        shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = recover_exports(args.audit, args.audit_sha256, args.out, check_only=args.check_only)
    print(json.dumps(result if args.check_only else {"recovered": True, "pages": len(result["pages"]),
        "results": str(args.out.resolve() / "results.json"), "replay_provider_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
