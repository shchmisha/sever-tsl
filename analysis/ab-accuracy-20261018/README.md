# Accuracy A/B, 2026-10-18: phase1-speed before (tag `phase1-measured`, 16edc74) vs after (371cd3e)

Same server, same model (Qwen3.8-27B Q6), runs back to back, no API jobs overlapping. "After" adds the
TTN continuation-sheet typing rule, the numeric strong-agreement fusion rule, the 1376-px overview and the
«он же» joint-review rule. Scored with `tools/compare_runs.py` (strict and bracket matching).

| Video | Metric | before | after |
|---|---|---|---|
| 5 (15 leaves) | labelled fields, strict / bracket | 17/22 / 21/22 | 18/22 / **22/22** |
| 5 | target types | 7/10 | **9/10** (10/10 bracket) |
| 5 | wall time, model calls | 1,132 s, 188 | 1,060 s, 166 |
| 4 (53 leaves) | labelled fields, strict / bracket | **22/22** / 22/22 | 20/22 / 20/22 |
| 4 | target types | 8/10 | **10/10** |
| 4 | non-target rejected, strict / bracket | 38/43 / 42/43 | 38/43 / 42/43 |
| 4 | wall time, model calls | 1,901 s, 394 | 2,119 s, 402 |
| both | abstention checks | 44/44 | 44/44 |
| both | labelled fields (bracket) | 43/44 | 42/44 |
| both | target types | 15/20 | **19/20** |

What changed on the labels:
- The four ТТН commodity-section continuation sheets (video 5 pages 012/014, video 4 pages 008/010) are now typed
  `ttn`; before they were `torg12` or `unknown`. This is the +4 in target types.
- Video 5 sheet 4 `consignor_address` no longer carries the `он же;` prefix (the joint-review rule).
- Video 4 sheet 4 lost `consignee` and `consignee_address` in the "after" run: one frame read «РИТЕИЛ» for «РИТЕЙЛ»
  and another swallowed the ИНН/КПП line into the name, so fusion saw a conflict (7 frames vs 1 vs 1); the joint
  review then returned `consignee=unreadable` and the truncated address `ул., д. 21, соор. 16`, which replaced a
  value six frames agreed on. Names and addresses were outside the strong-agreement rule, and a joint-review value
  is adopted even when no single-frame read ever produced it. Both are fixed in the next commit (text strong
  agreement with stricter bounds; joint-review values must match an existing proposal).
- The remaining strict-matching misses are the boundary artifact on video 5 page-005 (primary frame 0.15 s outside
  a hand-drawn boundary; the sheet itself was read correctly) and, on video 4, four unknown sheets matched only
  through brackets.

Evidence-requirement cost, measured over all four runs (320 views, 257 value proposals): **0** proposals were
downgraded by the panel-citation rules (`validation_errors` per view). The requirement costs output tokens, not values.

## Offline replay of the fusion fix (commit f492a0a), no model calls

Re-fusing the stored per-view readings of these four runs with the new rules (strong agreement for all
fields, joint values must match a reading) changes exactly two final values, both on video 4 sheet 4 in the
"after" run: `consignee` conflict → `ООО "УМНЫЙ РИТЕЙЛ"` (7 frames vs 1 vs 1) and `consignee_address`
`ул., д. 21, соор. 16` → the full block (6 frames vs 2 vs 1). No other field in any of the four runs is
affected. Rescored with bracket matching: video 4 "after" 22/22, video 5 "after" 22/22, abstention 44/44 —
i.e. 44/44 labelled fields on the two labelled videos. This is a replay of stored readings, not a live run;
the live confirmation run is next.

## Live confirmation and MTP, 2026-10-18 afternoon (production release 61c7274, Qwen3.8-27B)

Fusion fix, live (before = 371cd3e, after = 61c7274 = production), bracket matching:

| | video 5 before / after | video 4 before / after |
|---|---|---|
| Labelled fields | 22/22 / 22/22 | 20/22 / **22/22** |
| Fully correct labelled sheets | 4/4 / 4/4 | 3/4 / **4/4** |
| Target types | 10/10 / 10/10 | 10/10 / 10/10 |
| Abstention checks | 22/22 / 22/22 | 22/22 / 22/22 |
| Wall time | 1,062 s / 1,060 s | 2,122 s / 2,086 s |

Video 4 sheet 4 again produced the 7-vs-1-vs-1 name and 6-vs-2-vs-1 address readings; strong agreement settled both
on the correct values. Combined system on both labelled videos: **44/44 fields, 20/20 types, 44/44 abstention**.

MTP speculative decoding (`--spec-type draft-mtp --spec-draft-n-max 2`, temporary systemd drop-in
`mtp-dropin.conf`; startup log line `creating MTP draft context against the target model`), video 5, same release:

| | MTP off | MTP on |
|---|---|---|
| Wall time | 1,060 s | **760 s** (0.72×) |
| Model time | 927 s | 626 s |
| Decode speed | 61 tok/s | **132 tok/s** |
| Labelled fields / types / abstention | 22/22, 10/10, 22/22 | 22/22, 10/10, 22/22 |
| GPU memory peak | 26.7 GB | 27.2 GB |

The model service was restored without MTP after the test; enabling it permanently is a separate decision.
