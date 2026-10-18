# Labelling round 2, 2026-10-18

Five previously unlabelled videos (`20260821_144838`, `20260821_145110`, `20260818-6_131851`, `20260821_145342`,
`20260818_131702`; 54–101 s each) were labelled for the provisional ground truth
(`analysis/ground-truth-expanded-provisional.json`, videos with `labelling_round = "2026-10-18"`, ids `set2-<video>`).

| | count |
|---|---|
| visible leaves | 70 (26 non-target) |
| target leaves | 44: УПД 4, ТТН 18, ТОРГ-12 8, ТрН 14 |
| labelled values | 116 |
| labelled blanks (`not_present`) | 7 (ТрН «1а. Заказчик услуг…» left empty) |
| abstention checks (`not_visible_on_this_leaf`) | 60 |
| customer questions (`semantic_confirmation_needed`) | 15 (continuation-page №/date, ТрН sections with only an address, a signature-block date) |

Method (no output of the recognition system was consulted):
1. `tools/labelling/sample_frames.py` sampled each video every 0.5 s (effective 0.533 s) and built contact sheets.
2. Two independent labellers (`round/grouping-workflow.js`) split the samples into leaves and typed them; all 70 types
   agreed, boundaries agreed within one sample except two page-turn starts, settled by the assistant from the frames.
3. `tools/labelling/extract_tiles.py` cut native-resolution overlapping tiles (4×3 grid) from two frames per target leaf.
4. Two independent readers per leaf (`round/readers-workflow.js`, rules in `round/reader_rules.md`) transcribed every
   required field; a third reader settled disagreements (2 of 198 fields, both semantic questions). 196/198 fields
   were unanimous between the first two readers.
5. The assistant spot-checked values, blanks and nested quotes against the tiles (ТрН header/consignor/customer,
   ТОРГ-12 number/date/supplier); all matched.

Conventions match round 1: organisation fields hold the name only; УПД `consignee_address` is the entire printed
field; «он же» consignor_address is the seller's Адрес line; numbers and dates as printed. Values containing
look-alike letters (e.g. `РРРБ-128027`) are recorded as Cyrillic; `accepted_variants` may be needed if a reader of
Latin look-alikes should be accepted. `customer_approved=false` until a Север ТСЛ employee reviews the regenerated
`analysis/ground-truth-expanded-review.html`.

Score these videos with `tools/compare_runs.py` (any labelled video, matched by SHA-256) or with
`deploy/evaluate_server_results.py --include-round 2026-10-18`. Images (contact sheets, sampled frames, tiles)
are kept on disk and on the server under `/srv/sever/validation/labelling-20261018/`, not in git.
