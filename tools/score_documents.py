"""Score finished runs per document (form + own number) against the labelled videos.

Usage:
    python tools/score_documents.py --truth analysis/ground-truth-expanded-provisional.json \
        --out analysis/documents-20261018/score.json run1/results.json run2/results.json ...

Ground truth is read only after inference; runs are matched to labelled videos by
source SHA-256. Totals pool numerators and denominators across videos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sever.evaluation import evaluate, evaluate_documents, wilson_interval

KEYS = ("documents_found", "document_type_accuracy", "labeled_fields", "abstention_checks",
        "shown_values_correct", "e2e_on_fully_labeled")


def score(paths, truth_path):
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    rows = []
    for path in paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        video = next((item for item in truth["videos"] if item.get("source_sha256") == result["source"]["sha256"]), None)
        if video is None:
            rows.append({"run": str(path), "status": "no_labels_for_this_video"})
            continue
        documents = evaluate_documents(result, video)
        sheets = evaluate(result, video, matching="bracket")
        rows.append({"run": str(path), "video": video.get("source_id"), "status": "scored",
                     "elapsed_seconds": result.get("elapsed_seconds"),
                     "sheets_in_output": len(result.get("pages", [])),
                     "documents": {key: {k: documents[key][k] for k in ("correct", "total")} for key in KEYS},
                     "truth_documents": documents["truth_documents"], "output_documents": documents["output_documents"],
                     "extra_output_documents": documents["extra_output_documents"],
                     "abstention_failures": documents["abstention_checks"]["failures"],
                     "unscored_fields": documents["unscored_fields"],
                     "field_accuracy": documents["field_accuracy"],
                     "details": documents["details"],
                     "sheet_level": {"labeled_fields": {k: sheets["labeled_fields"][k] for k in ("correct", "total")}}})
    scored = [row for row in rows if row["status"] == "scored"]
    total = {key: {"correct": sum(row["documents"][key]["correct"] for row in scored),
                   "total": sum(row["documents"][key]["total"] for row in scored)} for key in KEYS}
    for value in total.values():
        value["rate"] = value["correct"] / value["total"] if value["total"] else None
        value["wilson95"] = wilson_interval(value["correct"], value["total"])
    fields = {}
    for row in scored:
        for name, value in row["field_accuracy"].items():
            fields.setdefault(name, {"correct": 0, "total": 0})
            fields[name]["correct"] += value["correct"]
            fields[name]["total"] += value["total"]
    return {"runs": rows, "total": total, "field_accuracy": fields,
            "truth_documents": sum(row["truth_documents"] for row in scored),
            "output_documents": sum(row["output_documents"] for row in scored),
            "extra_output_documents": sum(len(row["extra_output_documents"]) for row in scored)}


def markdown(report):
    names = {"documents_found": "Documents found", "document_type_accuracy": "Document type correct",
             "labeled_fields": "Labelled fields correct", "abstention_checks": "Fields correctly left empty",
             "shown_values_correct": "Values shown that are correct", "e2e_on_fully_labeled": "Documents fully correct"}
    scored = [row for row in report["runs"] if row["status"] == "scored"]
    lines = ["| | " + " | ".join(row["video"] for row in scored) + " | Total |", "|---|" + "---|" * (len(scored) + 1)]
    lines.append("| Documents labelled / output | " + " | ".join(f'{row["truth_documents"]} / {row["output_documents"]}' for row in scored)
                 + f' | {report["truth_documents"]} / {report["output_documents"]} |')
    for key, title in names.items():
        cells = [f'{row["documents"][key]["correct"]}/{row["documents"][key]["total"]}' for row in scored]
        total = report["total"][key]
        low = total["wilson95"][0] if total["wilson95"] else None
        lines.append(f"| {title} | " + " | ".join(cells) + f' | **{total["correct"]}/{total["total"]}**'
                     + (f" (95% CI from {low:.0%})" if low is not None else "") + " |")
    lines.append("| Sheets in output | " + " | ".join(str(row["sheets_in_output"]) for row in scored) + " | |")
    misses = []
    for row in scored:
        for detail in row["details"]:
            for name, ok in detail["field_checks"].items():
                if not ok:
                    misses.append(f'- {row["video"]} {detail["truth_document"]} {name}: output document '
                                  f'{detail["output_document"] or "none"}')
        for failure in row["abstention_failures"]:
            misses.append(f'- {row["video"]} {failure["truth_document"]} {failure["field"]}: value claimed where '
                          f'the field is not shown ({failure["raw_value"]!r})')
        for extra in row["extra_output_documents"]:
            misses.append(f'- {row["video"]} extra output document {extra["document_type"]} № {extra["number"]}: {extra["reason"]}')
    if misses:
        lines += ["", "Errors:", *misses]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = score(args.results, args.truth)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.out.with_suffix(".md").write_text(markdown(report) + "\n", encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
