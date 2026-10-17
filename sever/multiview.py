"""Adaptive source-view reading without a per-leaf frame-count limit."""
from __future__ import annotations

from pathlib import Path

from .aggregation import fields_needing_views, fields_to_read, merge_documents, neighbor_eligibility
from .agents import BatchBudgetExceeded, FatalProviderError
from .schema import DocumentResult


async def compare_page_views(source: Path, page: dict, ocr_pages: list[dict], document: DocumentResult,
                             expected_type: str, *, out: Path, local_ocr, reader, record,
                             additional_frames=None) -> DocumentResult:
    """Inspect retained views, then lazily obtain additional informative source PTS.

    Stop when fields are corroborated or explicitly disputed, or when there are
    no more usable same-leaf views. A shared monetary/token budget
    can interrupt model calls; this is disclosed, never presented as corroboration.
    """
    from .quality import assess_frame_quality
    from .frame_pool import iter_additional_page_frames

    comparison = {"initial_needed_fields": fields_needing_views(document, expected_type),
                  "attempts": [], "status": "not_needed", "frame_count_limit": None,
                  "stop_rule": "fields_corroborated_or_grounded_disputes_preserved_or_informative_views_exhausted"}
    page["neighbor_comparison"] = comparison
    if not comparison["initial_needed_fields"]:
        return document
    primary_ocr = ocr_pages[0]
    seen = {entry["frame_id"] for entry in ocr_pages}
    accepted = set(page["quality_gate"]["accepted_frame_ids"])
    retained = sorted((frame for frame in page["frames"] if frame["frame_id"] not in seen and frame["frame_id"] in accepted),
                      key=lambda frame: -frame.get("quality", {}).get("relative_selection_score", 0))
    pool_factory = additional_frames or iter_additional_page_frames
    pool = None

    def candidates():
        nonlocal pool
        yield from retained
        # The pool is not decoded/materialized if retained views already suffice.
        pool = iter(pool_factory(source, page, out/"additional-frames"))
        yield from pool

    candidates_iterator = candidates()
    comparison["status"] = "source_views_exhausted"
    try:
        while fields_to_read(document, expected_type):
            try:
                frame = next(candidates_iterator)
            except StopIteration:
                break
            if frame["frame_id"] in seen:
                continue
            seen.add(frame["frame_id"])
            needed = fields_to_read(document, expected_type)
            attempt = {"frame_id": frame["frame_id"], "timestamp_s": frame["timestamp_s"], "needed_fields": needed}
            comparison["attempts"].append(attempt)
            try:
                if frame["frame_id"] not in accepted:
                    quality = assess_frame_quality(frame)
                    attempt["quality_gate"] = quality
                    page["frames"].append(frame)
                    page["quality_gate"].setdefault("frame_dispositions", []).append(quality)
                    if not quality["accepted"]:
                        page["quality_gate"].setdefault("rejected_frames", []).append(quality)
                        attempt.update(status="skipped", reason="quality_rejected")
                        continue
                    accepted.add(frame["frame_id"])
                    page["quality_gate"]["accepted_frame_ids"].append(frame["frame_id"])
                neighbor_ocr = local_ocr({**frame, "ocr_storage_format": "jpeg"}, page["page_id"])
                ocr_pages.append(neighbor_ocr)
                eligibility = neighbor_eligibility(primary_ocr, neighbor_ocr, expected_type, needed)
                attempt["eligibility"] = eligibility
                if not eligibility["eligible"]:
                    attempt.update(status="skipped", reason=eligibility["reason"])
                    continue
                second = await reader([neighbor_ocr])
                second = DocumentResult.model_validate(second) if isinstance(second, dict) else second
                document, fusion = merge_documents(document, second, expected_type)
                attempt.update(status="compared", neighbor_result=second.model_dump(mode="json"), fusion=fusion)
            except FatalProviderError:
                raise
            except BatchBudgetExceeded:
                attempt.update(status="skipped", reason="shared_model_budget_exhausted")
                comparison["status"] = "shared_model_budget_exhausted"
                document.errors.append("multiview: shared_model_budget_exhausted; additional readings incomplete")
                break
            except Exception as exc:
                attempt.update(status="failed", error_type=type(exc).__name__)
            finally:
                record(attempt)
        if not fields_needing_views(document, expected_type):
            comparison["status"] = "all_fields_corroborated"
        elif not fields_to_read(document, expected_type):
            comparison["status"] = "grounded_conflicts_preserved"
        comparison["remaining_fields"] = fields_needing_views(document, expected_type)
    except FatalProviderError:
        raise
    except Exception as exc:
        comparison.update(status="additional_frame_pool_failed", error_type=type(exc).__name__)
        document.errors.append(f"multiview: additional_frame_pool_failed:{type(exc).__name__}")
        record({"status": "additional_frame_pool_failed", "error_type": type(exc).__name__})
    finally:
        comparison["remaining_fields"] = fields_needing_views(document, expected_type)
        candidates_iterator.close()
        if pool is not None and hasattr(pool, "close"):
            pool.close()
    return document
