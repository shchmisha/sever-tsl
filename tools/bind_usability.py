"""Bind independent frame reviews to unchanged manual leaf intervals, then evaluate.

No provider requests, labels, field values or interval boundaries are changed.
Unavailable primary frames count as availability failures, not illegible text.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sever.evaluation import evaluate
from sever.pipeline import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / "analysis/provisional-ground-truth.json"
    truth = json.loads(path.read_text())
    review = json.loads((root / "analysis/selected-frame-usability-full.json").read_text())
    indexed = {(e["source_sha256"], e["selected_frame_id"]): (i, e)
               for i, e in enumerate(review["entries"])}
    if len(indexed) != len(review["entries"]):
        raise ValueError("Duplicate source/frame review")
    inventories = json.dumps([v["pages"] for v in truth["videos"]], ensure_ascii=False, sort_keys=True)
    outputs = []
    for video in truth["videos"]:
        folder = root / "runs/mvp" / Path(video["source"]).stem
        result = json.loads((folder / "results.json").read_text())
        report = evaluate(result, video)
        annotations, used = {}, set()
        for detail in report["details"]:
            frame_id = detail["primary_frame_id"]
            if frame_id is None:
                annotations[detail["truth_page_id"]] = {
                    "selected_frame_id": None, "usable": False, "independently_reviewed": True,
                    "assessment": "primary_frame_availability", "provisional": True,
                    "reason": "No primary selected frame in this unchanged independently labeled leaf interval; not a judgment about text legibility.",
                }
                continue
            index, entry = indexed[(result["source"]["sha256"], frame_id)]
            with Path(entry["frame_path"]).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != entry["frame_sha256"]:
                raise ValueError("Reviewed frame bytes changed")
            used.add(index)
            annotations[detail["truth_page_id"]] = {
                "selected_frame_id": frame_id, "usable": entry["usable"],
                "independently_reviewed": entry["independently_reviewed"], "provisional": True,
                "assessment": entry["usability_scope"], "decision": entry["decision"],
                "required_fields_readable": entry["required_fields_readable"],
                "source_sha256": entry["source_sha256"], "frame_sha256": entry["frame_sha256"],
                "reasons": entry["reasons"],
                "review_reference": f"analysis/selected-frame-usability-full.json#/entries/{index}",
            }
        video["selected_frame_usability"] = annotations
        video["unmatched_selected_frame_review_indices"] = [i for i, e in enumerate(review["entries"])
            if e["source_sha256"] == result["source"]["sha256"] and i not in used]
        metrics = evaluate(result, {**video, "customer_approved": False, "provenance": truth.get("provenance")})
        save_json(folder / "metrics.json", metrics)
        outputs.append({"video": folder.name, "coverage": metrics["candidate_coverage"],
            "usable": metrics["frame_document_detection_rate"], "target_usable": metrics["frame_document_detection_rate_targets"],
            "fields_correct": sum(f["correct"] for f in metrics["field_accuracy"].values()),
            "fields_total": sum(f["total"] for f in metrics["field_accuracy"].values())})
    if inventories != json.dumps([v["pages"] for v in truth["videos"]], ensure_ascii=False, sort_keys=True):
        raise ValueError("Manual inventory changed")
    truth["usability_provenance"] = {
        "independent_review": "analysis/selected-frame-usability-full.json",
        "binding": "Exact source hash/frame ID and one-to-one primary timestamp matching to unchanged manual intervals.",
        "availability_misses": "No selected primary in a reviewed manual interval counts as an availability failure; nearby frames are not substituted.",
        "uncertain": "A null visual judgment remains uncertain; the confirmed-usable rate is a conservative lower bound.",
        "customer_approved": False,
    }
    save_json(path, truth)
    print(json.dumps(outputs, ensure_ascii=False))


if __name__ == "__main__":
    main()
