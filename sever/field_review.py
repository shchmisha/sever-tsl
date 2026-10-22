"""Bounded detail verification orchestration; original sheet readings stay intact."""
from __future__ import annotations

from pathlib import Path

from .agents import BatchBudgetExceeded, FatalProviderError
from .documents import assemble_documents
from .image_agent import EvidencePersistenceError
from .pipeline import trace
from .schema import FieldValue


def checked_detail_field(answer: dict, name: str, frames: list[dict]) -> FieldValue:
    if set(answer.get("fields", {})) != {name}:
        raise ValueError("Detail verification changed the requested field contract")
    field = FieldValue.model_validate(answer["fields"][name])
    allowed = {frame["frame_id"] for frame in frames}
    cited = {item.frame_id for item in field.evidence}
    if not cited <= allowed:
        raise ValueError("Detail verification cites another source frame")
    if field.status == "recognized":
        supporting = [frame for frame in frames if frame["frame_id"] in cited]
        if len(cited) < 2 or len({frame["source_sha256"] for frame in supporting}) < 2:
            raise ValueError("Detail verification requires two distinct source images")
    elif field.status == "not_present":
        # This pass is for disputed visible text. It must not turn a failed read
        # into a new assertion that the printed field is blank.
        raise ValueError("Disputed text verification cannot declare a field blank")
    elif field.raw_value is not None:
        raise ValueError("Unresolved detail verification cannot supply a value")
    return field


async def verify_document_fields(pages, *, source_id, config, out, trace_path, verifier, progress=None):
    """Review disagreements between sheets of an already assembled document.

    The assembly chooses membership before any calls. New detail readings never
    move a sheet to another document or overwrite its original reading.
    """
    from .verification_targets import frames_for_verification

    assembly = assemble_documents({"pages": pages})
    by_page = {page["page_id"]: page for page in pages}
    entries = []
    for record in assembly["documents"]:
        members = [by_page[sheet["page_id"]] for sheet in record["sheets"]]
        for name, original in record["fields"].items():
            candidates = original.get("candidates", [])
            # Truly absent/unreadable fields and single supported values need
            # no fine-print reread. A printed blank is not a text alternative.
            if (original.get("status") != "conflict" or len(candidates) < 2
                    or not all(item.get("status") == "recognized" for item in candidates)):
                continue
            frames = frames_for_verification(members, name)
            entry = {"source_id": source_id, "document_type": record["document_type"],
                "source_page_ids": sorted(page["page_id"] for page in members),
                "field_name": name, "trigger": "different_readings_on_sheets_of_one_document",
                "frame_ids": [frame["frame_id"] for frame in frames],
                "frame_pages": {frame["frame_id"]: frame["source_page_id"] for frame in frames},
                "status": "processing"}
            entries.append(entry)
            if len(frames) < 2:
                entry["status"] = "insufficient_source_views"
                entry["field"] = FieldValue(status="conflict",
                    reason="Distinct source views could not cover every disputed text reading").model_dump(mode="json")
                continue
            if progress:
                progress(f"Checking fine print across document sheets: {record['document_type']} / {name}")
            try:
                answer = await verifier(frames, source_id=source_id, page_id=record["document_id"],
                    document_type=record["document_type"], field_name=name, config=config,
                    cache_dir=Path(out) / "image-reading-evidence", trace_path=trace_path)
                field = checked_detail_field(answer, name, frames)
                entry.update(status="resolved" if field.status == "recognized" else "unresolved",
                             field=field.model_dump(mode="json"),
                             answer={**answer, "fields": {name: field.model_dump(mode="json")}})
            except (BatchBudgetExceeded, FatalProviderError, EvidencePersistenceError):
                entry["status"] = "failed"
                raise
            except Exception as exc:
                entry.update(status="failed", error_type=type(exc).__name__,
                    field=FieldValue(status="conflict", reason="Targeted detail verification did not confirm this field").model_dump(mode="json"))
            trace(trace_path, "document_field_detail_verification_finished", document_id=record["document_id"],
                  field_name=name, status=entry["status"], frame_ids=entry["frame_ids"])
    return entries
