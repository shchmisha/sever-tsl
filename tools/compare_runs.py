"""Compare two finished native_image runs of the same video: speed and labelled accuracy.

Works on the artifacts of any native_image version (results.json + trace.jsonl),
so a baseline release and a candidate release can be compared on the server.
Ground truth is read only here, after both recognition runs have finished.

    python tools/compare_runs.py --a RUN_A --label-a main --b RUN_B --label-b phase1-speed \
        --truth analysis/ground-truth-expanded-provisional.json \
        --out comparison.json --markdown comparison.md
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

# Each model request belongs to the most recent call kind that started on its leaf.
CALL_KINDS = {
    "image_orientation_observation": "orientation",
    "image_triage_observation": "triage",
    "image_agent_observation": "field_agent",
    "image_reader_observation": "single_read",
    "image_conflicts_observation": "joint_review",
}
TIMING_KEYS = ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms")


def _seconds(value):
    """Trace rows carry either epoch seconds (pipeline) or ISO-8601 text (model hooks)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    return None


def read_trace(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["_t"] = _seconds(row.get("time"))
            rows.append(row)
    return rows


def trace_stats(rows):
    """Per-leaf and total model calls, call kinds, model wall time and server timings."""
    leaves = defaultdict(lambda: {"requests": 0, "responses": 0, "kinds": Counter(), "model_seconds": 0.0,
                                  "input_tokens": 0, "output_tokens": 0, "first": None, "last": None})
    current_kind, pending = {}, {}
    timings = Counter()
    timed_responses = 0
    for row in rows:
        page = row.get("page_id")
        if page is None:
            continue
        leaf = leaves[page]
        moment = row["_t"]
        event = row.get("event")
        # Quality-gate rows for every leaf are written before any leaf is read;
        # a leaf's own time starts when it is selected for reading.
        if moment is not None and event == "page_selected":
            leaf["first"] = moment
        if moment is not None and leaf["first"] is not None:
            leaf["last"] = moment if leaf["last"] is None else max(leaf["last"], moment)
        if event in CALL_KINDS:
            current_kind[page] = CALL_KINDS[event]
        elif event == "model_request_started":
            leaf["requests"] += 1
            leaf["kinds"][current_kind.get(page, "unattributed")] += 1
            pending[page] = moment
        elif event == "model_response_received":
            leaf["responses"] += 1
            leaf["input_tokens"] += row.get("input_tokens") or 0
            leaf["output_tokens"] += row.get("output_tokens") or 0
            elapsed = row.get("elapsed_seconds")
            if elapsed is None and pending.get(page) is not None and moment is not None:
                elapsed = moment - pending[page]
            pending[page] = None
            leaf["model_seconds"] += elapsed or 0.0
            server = row.get("server_timings")
            if isinstance(server, dict):
                timed_responses += 1
                timings.update({key: server[key] for key in TIMING_KEYS if isinstance(server.get(key), (int, float))})
    per_leaf = {}
    for page, leaf in leaves.items():
        span = (leaf["last"] - leaf["first"]) if leaf["first"] is not None and leaf["last"] is not None else None
        per_leaf[page] = {"requests": leaf["requests"], "responses": leaf["responses"],
                          "call_kinds": dict(leaf["kinds"]), "model_seconds": round(leaf["model_seconds"], 2),
                          "leaf_seconds": round(span, 2) if span is not None else None,
                          "input_tokens": leaf["input_tokens"], "output_tokens": leaf["output_tokens"]}
    kinds = Counter()
    for leaf in per_leaf.values():
        kinds.update(leaf["call_kinds"])
    stamps = [row["_t"] for row in rows if row["_t"] is not None]
    totals = {"requests": sum(leaf["requests"] for leaf in per_leaf.values()),
              "responses": sum(leaf["responses"] for leaf in per_leaf.values()),
              "call_kinds": dict(kinds),
              "model_seconds": round(sum(leaf["model_seconds"] for leaf in per_leaf.values()), 2),
              "input_tokens": sum(leaf["input_tokens"] for leaf in per_leaf.values()),
              "output_tokens": sum(leaf["output_tokens"] for leaf in per_leaf.values()),
              "trace_started": min(stamps) if stamps else None, "trace_ended": max(stamps) if stamps else None}
    if timed_responses:
        totals["server_timings"] = {"responses": timed_responses, **{key: round(value, 2) for key, value in timings.items()}}
    return totals, per_leaf


def load_run(directory):
    directory = Path(directory)
    results_path = directory / "results.json"
    complete = results_path.is_file()
    results = json.loads((results_path if complete else directory / "partial-results.json").read_text(encoding="utf-8"))
    totals, per_leaf = trace_stats(read_trace(directory / "trace.jsonl"))
    pages = results.get("pages", [])
    for page in pages:
        recognition = page.get("image_recognition") or {}
        leaf = per_leaf.setdefault(page["page_id"], {"requests": 0, "responses": 0, "call_kinds": {},
            "model_seconds": 0.0, "leaf_seconds": None, "input_tokens": 0, "output_tokens": 0})
        fields = (page.get("result") or {}).get("fields") or {}
        leaf.update(interval_s=[page.get("start_s"), page.get("end_s")],
                    document_type=(page.get("result") or {}).get("document_type"),
                    stop_reason=recognition.get("stop_reason"), views=len(recognition.get("views") or []),
                    confirmed_fields=sum(field.get("status") in {"recognized", "not_present"} for field in fields.values()),
                    required_fields=len(fields))
    configuration = results.get("configuration") or {}
    summary = {"directory": str(directory), "complete": complete, "status": results.get("status", "completed" if complete else None),
               "source_sha256": (results.get("source") or {}).get("sha256"),
               "elapsed_seconds": results.get("elapsed_seconds"), "leaves": len(pages),
               "leaves_read": sum(bool((page.get("image_recognition") or {}).get("views")) for page in pages),
               "stop_reasons": dict(Counter((page.get("image_recognition") or {}).get("stop_reason") for page in pages)),
               "document_types": dict(Counter((page.get("result") or {}).get("document_type") for page in pages)),
               "native_pipeline_version": configuration.get("native_pipeline_version"),
               "leaf_concurrency": configuration.get("leaf_concurrency", 1),
               "trace": totals}
    if pages:
        summary["seconds_per_leaf"] = round(summary["elapsed_seconds"] / len(pages), 2) if summary["elapsed_seconds"] else None
        summary["requests_per_leaf"] = round(totals["requests"] / len(pages), 2)
    return results, summary, per_leaf


def score(results, truth_path, *, address_boundary="current", matching="strict"):
    """Score one run against the truth video with the same source SHA-256."""
    from sever.evaluation import evaluate

    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    sha = (results.get("source") or {}).get("sha256")
    video = next((video for video in truth.get("videos", []) if video.get("source_sha256") == sha), None)
    if video is None:
        return {"status": "no_labels_for_this_video"}
    report = evaluate(results, video, address_boundary=address_boundary, matching=matching)
    labelled = [value for value in report["field_accuracy"].values()]
    return {"status": "scored", "truth_video": video.get("source_id"), "address_boundary": address_boundary,
            "matching": matching, "bracket_matches": report.get("bracket_matches", []),
            "labelled_fields": {"correct": sum(v["correct"] for v in labelled), "total": sum(v["total"] for v in labelled)},
            "abstention_checks": {key: report["abstention_checks"][key] for key in
                                  ("correct", "total", "undetected_rows", "semantic_pending_rows", "failures")},
            "confidence_intervals": report.get("confidence_intervals", {}),
            "field_accuracy": {name: {"correct": value["correct"], "total": value["total"]}
                               for name, value in sorted(report["field_accuracy"].items())},
            "type_accuracy_targets": _count(report["type_accuracy_targets"]),
            "unknown_rejection": _count(report["unknown_rejection"]),
            "e2e_on_fully_labeled": _count(report["e2e_on_fully_labeled"]),
            "intervals_detected": _count(report["detection"]),
            "unmatched_predictions": report["unmatched_predictions"],
            "per_interval": [{key: detail.get(key) for key in ("truth_page_id", "prediction_page_id", "detected",
                              "type_correct", "field_checks", "e2e_success", "abstention_checks")} for detail in report["details"]]}


def paired_comparison(accuracy_a, accuracy_b):
    """Exact McNemar test on labelled fields and abstention checks both runs scored."""
    from sever.evaluation import mcnemar_exact

    result = {}
    for key in ("field_checks", "abstention_checks"):
        rows_a = {(item["truth_page_id"], name): ok for item in accuracy_a.get("per_interval", [])
                  for name, ok in (item.get(key) or {}).items()}
        rows_b = {(item["truth_page_id"], name): ok for item in accuracy_b.get("per_interval", [])
                  for name, ok in (item.get(key) or {}).items()}
        shared = sorted(set(rows_a) & set(rows_b))
        b = sum(rows_a[row] and not rows_b[row] for row in shared)
        c = sum(rows_b[row] and not rows_a[row] for row in shared)
        result[key] = {"paired_rows": len(shared), "a_only_correct": b, "b_only_correct": c,
                       "mcnemar_exact_p": mcnemar_exact(b, c),
                       "changed_rows": [{"truth_page_id": page, "field": name, "a": rows_a[(page, name)], "b": rows_b[(page, name)]}
                                        for page, name in shared if rows_a[(page, name)] != rows_b[(page, name)]]}
    return result


def _count(metric):
    return {"correct": metric["correct"], "total": metric["total"]}


def _ratio(new, old):
    return round(new / old, 3) if isinstance(new, (int, float)) and isinstance(old, (int, float)) and old else None


def compare(a_dir, b_dir, *, label_a="A", label_b="B", truth=None, address_boundary=("current", "current"), matching="strict"):
    runs = {}
    for label, directory, boundary in ((label_a, a_dir, address_boundary[0]), (label_b, b_dir, address_boundary[1])):
        results, summary, per_leaf = load_run(directory)
        accuracy = score(results, truth, address_boundary=boundary, matching=matching) if truth and summary["complete"] else (
            {"status": "run_incomplete_not_scored"} if truth else {"status": "no_truth_supplied"})
        runs[label] = {"summary": summary, "accuracy": accuracy, "per_leaf": per_leaf}
    a, b = runs[label_a]["summary"], runs[label_b]["summary"]
    paired = (paired_comparison(runs[label_a]["accuracy"], runs[label_b]["accuracy"])
              if all(runs[label]["accuracy"].get("status") == "scored" for label in (label_a, label_b)) else None)
    notes = []
    if a["source_sha256"] != b["source_sha256"]:
        notes.append("The two runs read different source videos; the comparison is not like for like.")
    if not (a["complete"] and b["complete"]):
        notes.append("At least one run did not finish; its numbers cover only the leaves it processed.")
    if a["leaves"] != b["leaves"]:
        notes.append("The runs segmented a different number of leaves; per-leaf rows are joined by page_id.")
    windows = [(run["summary"]["trace"]["trace_started"], run["summary"]["trace"]["trace_ended"]) for run in runs.values()]
    if all(start is not None and end is not None for start, end in windows):
        (s1, e1), (s2, e2) = windows
        if s1 < e2 and s2 < e1:
            notes.append("The two runs overlapped in time and competed for the model server; timings are not comparable.")
    notes.append("Timings include any other work on the model server during each run; check that no API jobs ran alongside.")
    speed = {"elapsed_seconds": _ratio(b["elapsed_seconds"], a["elapsed_seconds"]),
             "model_requests": _ratio(b["trace"]["requests"], a["trace"]["requests"]),
             "model_seconds": _ratio(b["trace"]["model_seconds"], a["trace"]["model_seconds"]),
             "input_tokens": _ratio(b["trace"]["input_tokens"], a["trace"]["input_tokens"]),
             "output_tokens": _ratio(b["trace"]["output_tokens"], a["trace"]["output_tokens"])}
    leaves = []
    for page in sorted(set(runs[label_a]["per_leaf"]) | set(runs[label_b]["per_leaf"])):
        row = {"page_id": page}
        for label in (label_a, label_b):
            leaf = runs[label]["per_leaf"].get(page) or {}
            row[label] = {key: leaf.get(key) for key in ("interval_s", "document_type", "stop_reason", "views",
                          "confirmed_fields", "required_fields", "requests", "model_seconds", "leaf_seconds", "call_kinds")}
        leaves.append(row)
    return {"labels": [label_a, label_b], "ratio_b_over_a": speed, "notes": notes, "paired": paired,
            "runs": {label: {"summary": run["summary"], "accuracy": run["accuracy"]} for label, run in runs.items()},
            "leaves": leaves}


def markdown(report):
    a, b = report["labels"]
    ra, rb = report["runs"][a], report["runs"][b]
    sa, sb = ra["summary"], rb["summary"]

    def cell(value):
        return "—" if value is None else f"{value:,.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else str(value)

    def frac(metric, interval=None):
        if not metric:
            return "—"
        text = f"{metric['correct']}/{metric['total']}"
        if interval:
            text += f" (95% {interval[0]*100:.0f}–{interval[1]*100:.0f}%)"
        return text

    lines = [f"# {a} vs {b}", "", f"| | {a} | {b} | {b}/{a} |", "|---|---|---|---|"]
    rows = [("Wall time, s", sa["elapsed_seconds"], sb["elapsed_seconds"], report["ratio_b_over_a"]["elapsed_seconds"]),
            ("Leaves", sa["leaves"], sb["leaves"], None),
            ("Seconds per leaf", sa.get("seconds_per_leaf"), sb.get("seconds_per_leaf"), None),
            ("Model requests", sa["trace"]["requests"], sb["trace"]["requests"], report["ratio_b_over_a"]["model_requests"]),
            ("Requests per leaf", sa.get("requests_per_leaf"), sb.get("requests_per_leaf"), None),
            ("Model time, s", sa["trace"]["model_seconds"], sb["trace"]["model_seconds"], report["ratio_b_over_a"]["model_seconds"]),
            ("Input tokens", sa["trace"]["input_tokens"], sb["trace"]["input_tokens"], report["ratio_b_over_a"]["input_tokens"]),
            ("Output tokens", sa["trace"]["output_tokens"], sb["trace"]["output_tokens"], report["ratio_b_over_a"]["output_tokens"])]
    lines += [f"| {name} | {cell(x)} | {cell(y)} | {cell(r)} |" for name, x, y, r in rows]
    kinds = sorted(set(sa["trace"]["call_kinds"]) | set(sb["trace"]["call_kinds"]))
    lines += [f"| Calls: {kind} | {cell(sa['trace']['call_kinds'].get(kind, 0))} | {cell(sb['trace']['call_kinds'].get(kind, 0))} | |" for kind in kinds]
    if ra["accuracy"].get("status") == "scored" or rb["accuracy"].get("status") == "scored":
        boundaries = [run["accuracy"].get("address_boundary", "current") for run in (ra, rb)]
        matching = ra["accuracy"].get("matching", "strict")
        lines += ["", f"| Accuracy (provisional labels; matching {matching}; address boundary {a}: {boundaries[0]}, {b}: {boundaries[1]}) | " + a + " | " + b + " |", "|---|---|---|"]
        intervals = {"labelled_fields": "labeled_fields", "type_accuracy_targets": "type_accuracy_targets",
                     "unknown_rejection": "unknown_rejection", "e2e_on_fully_labeled": "e2e_on_fully_labeled",
                     "intervals_detected": "detection", "abstention_checks": "abstention_checks"}
        for name, key in (("Labelled fields correct", "labelled_fields"), ("Target types correct", "type_accuracy_targets"),
                          ("Non-target rejected", "unknown_rejection"), ("Fully correct labelled sheets", "e2e_on_fully_labeled"),
                          ("Labelled intervals detected", "intervals_detected"),
                          ("Abstention checks passed (not-visible rows)", "abstention_checks")):
            cells = [frac(run["accuracy"].get(key), (run["accuracy"].get("confidence_intervals") or {}).get(intervals[key]))
                     for run in (ra, rb)]
            lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
        if report.get("paired"):
            for key, title in (("field_checks", "labelled fields"), ("abstention_checks", "abstention checks")):
                pair = report["paired"][key]
                p_value = pair["mcnemar_exact_p"]
                lines.append(f"| Paired {title}: {a}-only / {b}-only correct | {pair['a_only_correct']} / {pair['b_only_correct']} of {pair['paired_rows']} | "
                             f"McNemar p = {'—' if p_value is None else f'{p_value:.3f}'} |")
        failures = [(label, f) for label, run in ((a, ra), (b, rb)) for f in (run["accuracy"].get("abstention_checks") or {}).get("failures", [])]
        if failures:
            lines += ["", "Abstention failures (value claimed for a field marked not visible):",
                      *[f"- {label}: {f['truth_page_id']} {f['field']} → {f['status']} {f.get('raw_value') or ''}".rstrip() for label, f in failures]]
        changed = []
        for item_a, item_b in zip(ra["accuracy"].get("per_interval", []), rb["accuracy"].get("per_interval", [])):
            for field in sorted(set(item_a.get("field_checks") or {}) | set(item_b.get("field_checks") or {})):
                left, right = (item_a.get("field_checks") or {}).get(field), (item_b.get("field_checks") or {}).get(field)
                if left != right:
                    changed.append(f"- {item_a['truth_page_id']} {field}: {a}={'ok' if left else 'wrong'}, {b}={'ok' if right else 'wrong'}")
        if changed:
            lines += ["", "Fields that changed:", *changed]
        brackets = [(label, m) for label, run in ((a, ra), (b, rb)) for m in run["accuracy"].get("bracket_matches", [])]
        if brackets:
            lines += ["", "Matched through a boundary bracket (check by hand):",
                      *[f"- {label}: {m['truth_page_id']} primary at {m['primary_timestamp_s']:.2f} s, core {m['core_interval_s'][0]:.2f}–{m['core_interval_s'][1]:.2f} s"
                        for label, m in brackets]]
    else:
        lines += ["", f"Accuracy: {ra['accuracy'].get('status')} / {rb['accuracy'].get('status')}"]
    lines += ["", "| Leaf | " + a + " requests | " + b + " requests | " + a + " s | " + b + " s | " + a + " stop | " + b + " stop |",
              "|---|---|---|---|---|---|---|"]
    for row in report["leaves"]:
        x, y = row[a], row[b]
        lines.append(f"| {row['page_id']} | {cell(x.get('requests'))} | {cell(y.get('requests'))} | {cell(x.get('leaf_seconds'))} | "
                     f"{cell(y.get('leaf_seconds'))} | {x.get('stop_reason') or '—'} | {y.get('stop_reason') or '—'} |")
    if sb["trace"].get("server_timings"):
        timing = sb["trace"]["server_timings"]
        lines += ["", f"{b} server timings over {timing['responses']} responses: cached {cell(timing.get('cache_n'))}, "
                  f"processed {cell(timing.get('prompt_n'))} input tokens in {cell(timing.get('prompt_ms'))} ms; "
                  f"generated {cell(timing.get('predicted_n'))} tokens in {cell(timing.get('predicted_ms'))} ms."]
    lines += ["", *[f"- {note}" for note in report["notes"]]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--a", required=True, type=Path, help="Baseline run directory")
    parser.add_argument("--b", required=True, type=Path, help="Candidate run directory")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--truth", type=Path, help="Ground-truth JSON with videos[].source_sha256 (read after inference)")
    parser.add_argument("--address-boundary-a", choices=["current", "legacy"], default="current",
                        help="Score run A's address fields against legacy_value (pre-2026-10-11 postal-only labels)")
    parser.add_argument("--address-boundary-b", choices=["current", "legacy"], default="current")
    parser.add_argument("--matching", choices=["strict", "bracket"], default="strict",
                        help="bracket: a primary frame inside a label's boundary bracket may match it (reported separately)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    if args.label_a == args.label_b:
        parser.error("Labels must differ")
    report = compare(args.a, args.b, label_a=args.label_a, label_b=args.label_b, truth=args.truth,
                     address_boundary=(args.address_boundary_a, args.address_boundary_b), matching=args.matching)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(markdown(report), encoding="utf-8")
        print(args.markdown.read_text(encoding="utf-8"))
    return report


if __name__ == "__main__":
    main()
