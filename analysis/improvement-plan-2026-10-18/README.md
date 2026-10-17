# Improvement research, 2026-10-18

`plan.md` is the synthesized speed + accuracy plan. It was produced by a 25-agent workflow:
5 code readers (segmentation, reading step, fusion/stop policy, serving config, evaluation/labels),
6 web researchers (general VLMs, OCR specialists, serving engines, techniques, Russian OCR evidence,
cloud yardstick), one adversarial verifier per report (`verification.json`: 102 confirmed,
29 uncertain, 10 refuted claims; refuted claims were dropped before synthesis), a synthesis pass,
a completeness critic (`critique.md`) and a revision applying the critique. `reports.json` holds the
surviving findings and model candidates with their evidence links.

Caveats that matter: model cards were not opened by the verifiers (rows marked †); llama.cpp behaviour
was checked against a local b7822 checkout while the server runs a newer "v0.6.0" archive; all accuracy
numbers rest on 44 provisional labels. Nothing in the plan has been implemented or measured yet unless
it says "measured".

## Server facts checked after the plan was written (2026-10-18, read-only)

- The deployed `llama-server --help` lists `--spec-type none,draft-simple,draft-eagle3,draft-mtp,...`
  and `--spec-draft-n-max N (default: 3)`, so MTP self-speculation (plan item C2 / §5) is available in this build.
- The deployed GGUF `Qwen3.8-27B-UD-Q6_K_M.gguf` contains the MTP tensors: the startup log of the current
  `sever-model` run (started 2026-10-10 19:21 MSK) says `model has unused tensor blk.64.nextn.eh_proj.weight
  (size = 43008000 bytes) -- ignoring`, plus `enorm`, `hnorm`, `shared_head_norm`. They are loaded and unused.
- The RAM prompt cache is active with the mmproj loaded (contrary to the b7822 assumption in the plan):
  the log shows `making room for prompt cache entry, removing oldest entry (size ≈ 1.18–1.81 GiB)` about once
  a minute during runs. Each cached context is ~1.75 GiB, so the default budget holds only a few leaves;
  a larger `--cache-ram` is a cheap thing to test alongside M2.
- GPU memory during the A/B stayed at 26.2–26.7 GB of 32.6 GB, so the plan's ≤29.5 GB MTP budget has ~6 GB of room.
- Still unverified: whether MTP leaves grammar-constrained tool output unchanged at temperature 0, and the
  actual decode gain on this image-heavy workload. Both need the §5 measurement (two MTP-off runs, then MTP-on).
