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
