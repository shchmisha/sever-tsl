# A/B speed test, 2026-10-17: `main` (V7 code) vs `phase1-speed`

Server `ai-ocr`, RTX 5090, Qwen3.8-27B-UD-Q6_K_M via llama-server (`--parallel 1`, ctx 65536).
Video `20260818-5_132107.mp4` (72.8 s, 15 detected leaves), one run after the other, no API jobs overlapping
(`run-*.json: api_jobs_active_at_end=0`). Both releases staged from the git bundle at
`/opt/sever-tsl/releases/20261017-191924-ab-{a,b}`; tests passed on the server (1304 and 1342 passed, 4 skipped).
Raw runs: `/srv/sever/validation/ab-speed-20261017-191924/`. `comparison.json`/`.md` here were produced by
`tools/compare_runs.py` at f52a8e2; `gpu-*.csv` is `nvidia-smi` every 5 s during each run.

| | main | phase1-speed | ratio |
|---|---|---|---|
| Wall time | 2,665 s (44.4 min) | 1,134 s (18.9 min) | **0.43** |
| Seconds per leaf | 177.7 | 75.6 | 0.43 |
| Model requests | 424 | 188 | 0.44 |
| Model time | 2,411 s (90% of wall) | 1,004 s (89%) | 0.42 |
| Seconds per request | 5.69 | 5.34 | |
| Input tokens per request | 11,396 | 10,596 | |
| Output tokens per request | 236 | 222 | |
| Calls by kind | orientation 75, field agent 294, joint review 55 | orientation 15, triage 19, field agent 115, joint review 39 | |
| GPU memory (min–max) | 26,180–26,683 MiB | same | |

Where the time went (phase1-speed, from llama.cpp `timings` per response): prefill 639,726 new input tokens
in 273 s (2,340 tok/s); **generation 41,696 tokens in 688 s (61 tok/s) = 69% of model time**; 68% of input
tokens came from the prefix cache. Reducing output tokens or decoding faster (batched decode, a faster-decoding
model) is now the main lever per call; the number of calls is the lever per sheet.

Per leaf (seconds from the leaf's `page_selected` event):

| Leaf | kind / stop | main calls | p1 calls | main s | p1 s |
|---|---|---|---|---|---|
| 5 out-of-scope leaves (0007–0011) | out_of_scope_form_confirmed | 7–8 each | 3 each (one 8) | 23–40 | 6 (one 21) |
| 4 clean target leaves (0001, 0002, 0004, 0005) | all_required_fields_confirmed | 9–13 | 8–15 | 52–76 | 31–59 |
| 2 TTN leaves with fields on the other side (0003, 0006) | requires_other_document_side | 18–22 | 19–22 | 119–121 | 105–109 |
| 4 hard leaves (0012–0015: torg12 ×2, upd ×2) | other side / exhausted | 44–106 | 13–34 | 256–776 | 97–301 |

Accuracy on the provisional labels for this video (22 labelled fields): **main 16/22, phase1-speed 15/22**;
target types 7/10 both; non-target rejection 5/5 both; fully correct labelled sheets 1/4 both;
labelled intervals detected 14/15 both. The one differing field is `sample-05-page-004 consignor_address`:
on phase1-speed the two frame reads differed by one glyph («Кашириных» vs «Каширных»), the joint review then
returned the whole printed block `он же; ООО "ФРЭШ-ЭКСПРЕСС", 454004, …` where the label holds the postal
part only (label follows the pre-11.10 field boundary). main happened to get two identical reads. This is
label-boundary ambiguity plus one-frame model noise, not a change in reading behaviour; with 22 labels the
95% interval of either score spans roughly 50–87%.

Not covered by this test: `HARNESS_LEAF_CONCURRENCY>1` (server runs `--parallel 1`), the other 10 videos,
and any customer-approved labels.

## Rescored on 2026-10-18 with the relabelled addresses and the new evaluator

`comparison-relabelled.md` / `.json` rescore the same two runs after the four `consignee_address` labels were
moved to the entire printed block (2026-10-11 rule; postal-only text kept as `legacy_value`) and after the
evaluator gained abstention checks and Wilson intervals (commit after 5131285):

| | main | phase1-speed |
|---|---|---|
| Labelled fields correct | 18/22 (95% 61–93%) | 17/22 (95% 57–90%) |
| Fully correct labelled sheets | 3/4 | 2/4 |
| Abstention checks (22 not-visible rows on detected sheets) | 22/22 | 22/22 |
| Paired difference | 1 field (main-only correct), McNemar p = 1.0 | |

Scored against the legacy postal-only labels, main reads 16/22; the two consignee addresses it read in full were
correct all along. The one remaining difference is the `он же;`-prefixed joint-review value on sheet 4, which the
joint-review instruction now forbids (unmeasured until the next live run).
