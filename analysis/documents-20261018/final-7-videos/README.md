# Final measurement on all 7 labelled videos, 2026-10-18/19

Production release `20261018-majority` (commit 8519b29) as customers use it. Each labelled video was read as an
ordinary customer-cabinet job (team `sever-test`, portal settings: sample interval 0.5 s, up to 300 sheets, fresh
cache), one job at a time, with no other GPU work alongside. The ground truth
(`analysis/ground-truth-expanded-provisional.json`, provisional, not customer-approved) was opened only after each
job finished. The Russian final report built on these numbers is `analysis/final-report-20261018/README.md`.

| Video | Truth id | Portal job | API job (results in `/srv/sever/runs/<id>/`) | Sheets out | Seconds |
|---|---|---|---|---|---|
| 20260818-4_132351 | sample-04 | d8250d01-2bb4-4faf-9eea-3bec361984c3 | 61eead11-6868-4275-8739-0665b0a9d04d | 56 | 1,089 |
| 20260818-5_132107 | sample-05 | 18c12a7d-04d4-407b-9a2d-d9e66dfd8f38 | 0ae91181-eb6e-4196-89b9-73b4dcb2a184 | 15 | 605 |
| 20260821_144838 | set2-20260821_144838 | 66f59852-25ac-4e63-9154-2964c893c5d3 | c47131cf-8c2c-456e-9413-ddbbe497e381 | 12 | 649 |
| 20260821_145110 | set2-20260821_145110 | bb970345-e34c-40e8-ba38-62aa751f9f7b | 2de71c2b-9073-4a53-96cc-9355763cca16 | 36 | 2,351 |
| 20260818-6_131851 | set2-20260818-6_131851 | 0856b935-fa7e-45d4-a85d-c145709e930f | d8ea7ae0-76a4-4de0-a90f-691487c148cc | 25 | 1,133 |
| 20260821_145342 | set2-20260821_145342 | c728276c-a3cf-419b-b4dc-cb1adc8cbe7b | f3641a0b-e052-4d68-8ae8-5e16a12e8276 | 31 | 2,403 |
| 20260818_131702 | set2-20260818_131702 | 593cd880-3b7d-4fc3-8689-ebebfda4c4ed | 943548f5-e539-4bb6-a821-ac144f5e5972 | 17 | 544 |

## Results

| | Per document | Per sheet showing |
|---|---|---|
| Documents found / target sheets with a selected frame | 17/18 | 66/67 |
| Type correct | 17/18 | 60/67 |
| Labelled fields correct | 70/75 | 159/173 |
| Values shown that are correct | 67/68 | |
| Fully labelled documents / sheets fully correct | 12/14 | 24/31 |
| Non-target sheets kept out of the 4 types | | 67/71 |
| Extra output documents | 4 of 21 | |

Against the earlier per-document result (`../score-4-videos.md`, release 61c7274: 38/39 fields, 7/8 documents fully
correct) the same 4 videos give 38/39 and 7/8 again: the ТрН number in 145110 is now right (majority rule) and the
УПД consignee in sample-04 is now lost (one sheet read «РИТЕИЛ» without the breve, a 1:1 tie). Extra documents on
those videos went from 1 to 3. One of the two new ones is a regression: a route sheet (маршрутный лист) whose title is
hidden under a thumb is now typed ТТН on 7 of 8 views, where 61c7274 said non-target on 8 of 9 views; 145342 shows
the same. Per-sheet A/B on the 5 videos with an earlier run: `ab-vs-earlier/` (labelled-field changes ≤ 2 per video,
McNemar p = 1.0; wall time 0.52–0.96×).

## Files

- `score-7-videos.{json,md}` — `tools/score_documents.py` on the 7 cabinet results.
- `sheets-7-videos.{json,md}` — `tools/compare_runs.py` `score(..., matching="bracket")` per video, pooled.
- `misses-7-videos.json` — every document- and sheet-level miss with truth vs output values and the frames used.
- `earlier-readings-current-truth.{json,md}` — the earlier (61c7274) readings rescored against today's truth
  (38/41: the 145110 счёт-фактура was labelled as УПД on 18.10).
- `ab-vs-earlier/` — `tools/compare_runs.py --matching bracket`, earlier run vs cabinet run, 5 videos.
- `tables/<video>/documents.{csv,json}` — the per-document export of each cabinet run.

Error causes (frames inspected for every miss) are in `analysis/final-report-20261018/error-statistics.{md,json}`.

Reproduce (results fetched from `/srv/sever/runs/<API job>/results.json`):

```bash
uv run python tools/score_documents.py --truth analysis/ground-truth-expanded-provisional.json --out score-7-videos.json RESULTS...
```
