"""Compare saved baseline/final results against the SAME independent labels.

Read-only with respect to source results and reference labels. No provider calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sever.evaluation import evaluate
from sever.pipeline import save_json
from build_delivery import add_ratio, file_hash, relative


def totals(metrics):
    names = ("candidate_coverage", "type_accuracy_targets", "unknown_rejection", "e2e_on_fully_labeled")
    result = {name: add_ratio([item[name] for item in metrics]) for name in names}
    result["labeled_fields"] = add_ratio([value for item in metrics for value in item["field_accuracy"].values()])
    return result


def build(root, out, baseline, truth_path):
    truth = json.loads(truth_path.read_text())
    previous, current, rows = [], [], []
    states, skips, gates, registries = Counter(), Counter(), Counter(), Counter()
    recovered, conflicts, agreements, partial = [], [], [], []
    frame_counts = []
    for expected in truth["videos"]:
        stem = Path(expected["source"]).stem
        old_path, final_path = baseline/stem/"results.json", out/stem/"results.json"
        old = json.loads(old_path.read_text())
        final = json.loads(final_path.read_text())
        expected = {**expected, "customer_approved": False, "provenance": truth.get("provenance")}
        before, after = evaluate(old, expected), evaluate(final, expected)
        previous.append(before); current.append(after)
        rows.append({"source": final["source"], "baseline": before, "final": after,
            "result_inputs": {"baseline": {"path": relative(old_path, root), "sha256": file_hash(old_path)},
                              "final": {"path": relative(final_path, root), "sha256": file_hash(final_path)}}})
        for page in final["pages"]:
            views = page.get("neighbor_comparison", {})
            if views:
                states[views.get("status", "unknown")] += 1
                frame_counts.append({"source":stem, "page_id":page["page_id"],
                    "retained_frames":len(page["frames"]), "additional_views_considered":len(views.get("attempts",[])),
                    "model_views_compared":sum(a.get("status")=="compared" for a in views.get("attempts",[])),
                    "remaining_fields":views.get("remaining_fields",[])})
            for attempt in views.get("attempts", []):
                if attempt.get("status") != "compared":
                    skips[attempt.get("reason") or attempt.get("error_type") or attempt.get("status", "unknown")] += 1
                fusion = attempt.get("fusion", {})
                for key, destination in (("recovered_fields",recovered),("conflicting_fields",conflicts),("agreement_fields",agreements)):
                    destination.extend({"source":stem,"page_id":page["page_id"],"frame_id":attempt["frame_id"],"field":field}
                                       for field in fusion.get(key,[]))
                partial.extend({"source":stem,"page_id":page["page_id"],**value} for value in fusion.get("partial_observations",[]))
            for disposition in page.get("quality_gate",{}).get("frame_dispositions",[]):
                gates[disposition["status"]] += 1
            for value in page.get("inn_verification",{}).get("roles",{}).values():
                registries[value.get("registry",{}).get("status","unknown")] += 1
    snapshot = {"date":"2026-10-07", "customer_approved":False,
        "truth_sha256":hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        "comparison_rule":"Both versions evaluated against identical independent provisional labels; no baseline outputs rewritten",
        "baseline":totals(previous),"final":totals(current),"by_source":rows,
        "multiview":{"frame_count_limit":None,"page_states":dict(states),"skip_reasons":dict(skips),
                     "recovered_fields":recovered,"conflicting_fields":conflicts,"agreement_fields":agreements,
                     "partial_observations":partial,"frame_counts":frame_counts},
        "quality_dispositions":dict(gates),"inn_registry_statuses":dict(registries),
        "acceptance_owner":"Customer employee reviews reference values first; measured results remain provisional"}
    save_json(out/"baseline-comparison.json",snapshot)
    return snapshot


if __name__ == "__main__":
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=root/"runs/closing")
    parser.add_argument("--baseline",type=Path,default=root/"runs/mvp")
    parser.add_argument("--truth",type=Path,default=root/"analysis/ground-truth-expanded-provisional.json")
    args=parser.parse_args()
    result=build(root,args.out.resolve(),args.baseline.resolve(),args.truth.resolve())
    print(json.dumps({"baseline":result["baseline"],"final":result["final"],"states":result["multiview"]["page_states"]},ensure_ascii=False))
