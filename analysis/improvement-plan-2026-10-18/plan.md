# Север: speed and accuracy plan (as of 2026-10-18)

Audience: the engineer who owns the pipeline. All numbers come from the attached verified reports. When this plan says "measured", it means one of these runs: the 2026-10-17 server A/B (`analysis/ab-speed-20261017/`, commit 953e8e3), the V7 batch, or local CPU replays. Anything else is labelled as an estimate (est.) or unverified (†).

## 0. Corrections to the brief

These change the priorities, so they come first.

1. **Phase 1 has already been measured.** See `analysis/ab-speed-20261017/README.md` and `comparison.md`/`comparison.json`. On video 5:
   - wall time went from 2665 s to 1134 s;
   - calls went from 424 to 188;
   - labelled fields went from 16/22 to 15/22.

   The one regression is page-0004 consignor_address, where the joint review returned «он же; ООО "ФРЭШ-ЭКСПРЕСС", …».
2. **"8/20 target intervals missed" is a mislabelled metric.** `missed_intervals: 8` counts all 88 labelled intervals (`sever/evaluation.py:203`). `tools/compare_runs.py:162` displays it at `:242` as "Labelled intervals detected".
   - The evaluator only credits a type when the interval was detected (`evaluation.py:111-117`). V7 typed 14 targets correctly, so V7 missed at most 6 targets.
   - A replay of the strict matcher suggests 1 target miss: sample-05-page-005. That miss is a boundary artifact. The primary frame is at 21.32 s, the label ends at 21.174 s, and the label's own end bracket is [20.658, 21.691] (truth file line 4413).
   - The V7 segmentation itself was not replayed. Confirm this against `/root/sever-validation/native-v7-evaluation/finish-evaluation.json` before publishing "1 of 20".
3. **Segmentation is therefore not the largest accuracy loss on labelled data.** On video 5 the losses are:
   - 2 consignee_address labels that use the pre-11.10 postal-only boundary. Output that follows the current rule is scored wrong.
   - 4 fields lost to the page-005 boundary artifact.
   - The TTN continuations (pages 012/014) typed as ТОРГ-12. This happens in both arms.
   - The «он же» joint-review value.
4. **The workload is decode-bound.** Phase 1 per call (`README.md:22-24`):

   | Part | Time | Detail |
   |---|---|---|
   | Prefill | 1.45 s | 3,403 new tokens at 2,340 tok/s |
   | Decode | 3.66 s | 222 tokens at 61 tok/s |
   | Other | 0.23 s | |

   Decode is 69% of model time, and 68% of input tokens came from the prefix cache.
5. **llama.cpp behaviour was mostly checked against the local b7822 checkout (`~/llama.cpp`).** The server runs a "v0.6.0" tag archive (`deploy/README.md:254,259`). It is newer: it accepts `--no-reasoning-preserve`, and b7822 does not.
   - When an mmproj is loaded, b7822 refuses to start speculative decoding. It also disables `--cache-reuse`, the RAM prompt cache and context checkpoints.
   - C2 (MTP) and M2 (checkpoints) therefore work only if v0.6.0 behaves differently.
   - Check every flag with `llama-server --help` and the startup log on the server first.
6. **The brief misdescribes the earlier MoE trial.** `deploy/README.md:236-246` describes it as follows:
   - It covered **two** short clips, not one.
   - It ran on the OCR → classification → reading pipeline from before the v6 fix.
   - Its "9/10" figures count fields "accepted by the evidence check, not independently measured accuracy".

## 1. Baseline scorecard

| Metric | V7 batch (11 videos, main) | Video 5, main (V7 code) | Video 5, phase1-speed | Realistic target |
|---|---|---|---|---|
| Seconds per sheet | 174 | 177.7 (2665/15) | 75.6 (1134/15) | 40–55 single-stream on video 5 (est.). ≤35 batch throughput (est.): phase1 has never run on the 11-video batch, so that baseline comes first (P0). |
| Calls per sheet | ~34 (9222/273) | 28.3 (424/15) | 12.5 (188/15) | ≤8 |
| Seconds per call | not split | 5.69 (2411 s / 424) | 5.34 (1.45 prefill / 3.66 decode / 0.23 other) | ~3.7–4.3 with MTP (est.); ~2 with the MoE (est.) |
| Decode tok/s | — | — | 61 | ≥90 with MTP (est.). The only vision+MTP report is ~92 tok/s, measured at n-max 5. |
| Labelled-field accuracy (correct/labelled) | 35/44 = 79.5% (Wilson 95% CI 65.5–88.8) | 16/22 (CI 52–87) | 15/22 (CI 47–84) | After the consignee_address relabel, main should read ~18/22. With bracket matching it could reach 22/22, if the page-005 values match the labels (team Q8). Target: non-inferior to main (≤1 net lost field on 44, and no new false values on the abstention checks). ±5-point resolution needs ~250 labels. |
| Target-type accuracy | 14/20 | 7/10 | 7/10 | 10/10. This needs page-005 recovered (M8 or brackets) and both TTN continuations no longer typed as ТОРГ-12 (M11). |
| Missed target intervals, strict | ≤6, likely 1 (unconfirmed) | 1/10 (page-005) | 1/10 | 0/10 strict (primary-frame centrality); 0 with brackets |
| Missed intervals, all | 8/88 | 1/15 | 1/15 | ≤3/88 strict (speculative) |
| Fully right labelled sheets | 4/8 | — | — | report together with a CI |
| Abstention checks (not_visible rows) | not scored | not scored | not scored | report; no false values or roles |
| GPU memory in use | — | — | 26.2–26.7 of 32.6 GB | ≤29.5 GB with MTP |

The accuracy CIs are wide. With 22 labels, a difference of 1–3 fields is noise. On video 5 alone, an exact McNemar test needs at least 6–0 flips to reach p < 0.05.

## 2. Bottleneck inventory

Status labels:
- **V**: verified by a report's verify pass.
- **U**: uncertain, flagged in the report.
- **†**: web finding with no verify pass.

**Precision gate for C1, M6, m2 and m3.** These items act only on leaves whose labels are all `not_visible_on_this_leaf`, and `evaluation.py` skips those rows. "Labelled ≥15/22" therefore cannot detect a regression on them. Each of these items is gated on the abstention check added in C4: no false values or roles on the not_visible rows.

On video 5, only the 8 not_visible rows on the two correctly typed UPD continuations (013/015) carry signal. The two TTN continuations (012/014) already fail on type in both arms.

### Critical

| # | Bottleneck (measured now) | Change | Estimated improvement | Effort | Risk | How to test on the server | Status |
|---|---|---|---|---|---|---|---|
| C1 | **Leaves stopping on "other document side".** In phase 1 these 6 leaves (0003, 0006, 0012–0015) take 122/188 calls, ~821/1134 s (72%) and 33/39 joint-review calls. Their confirmed fields are identical in main and phase 1. The only route to `requires_other_document_side` is a joint review (`native_pipeline.py:672-680, :821-823`; `image_conflicts.py:330-336`). `ImageField` has only a free-text `reason` (`image_reader.py:159-167`). | Add an optional per-field enum `visibility_cause`, allowed only with status unreadable. Accept the other-side stop when 2 distinct views, each with `complete_page_visible=True`, both say other_document_side. In that case skip `review_conflicts("initial")`. Keep every existing guard: no value or blank proposals, no conflicts, no pending type review. | Realistic upper bound −280 to −340 s on video 5, from the 5 leaves that stopped at view 2. About −200 s if only half the leaves flag it. Do not count 0015's ~137 s share. | M | Changes the design rule that only a joint review may declare that this side lacks a field. The model may overuse the flag on blurred frames. Requires updates to `tests/test_native_conflict_pipeline.py`. | (1) Offline, no calls: in the phase1 run under `/srv/sever/validation/ab-speed-20261017-191924/`, compare per-view `reason` strings on the 6 leaves with the joint review's `visibility_cause`. (2) Run the `deploy/ab_speed_test.sh` A/B. Targets: joint calls on these leaves 33 → ≤10; their seconds ≤550; labelled fields ≥15/22; no false values or roles on the not_visible rows (abstention gate). | V (gain unmeasured) |
| C2 | **Decode-bound at 61 tok/s, 69% of model time.** `deploy/sever-model.service:12` has no `--spec-type`. b7822 refuses speculative decoding when an mmproj is loaded (serving report, verified). C2 therefore depends entirely on v0.6.0 being newer than b7822 and containing #28715 (v0.4.1, "Fixed speculative decoding after image input"). The engines report names #28715 as the fix for MTP+vision. | Enable MTP self-speculation: `--spec-type draft-mtp --spec-draft-n-max 2`. Sweep 1–3; the default is 3. Keep `--parallel 1`. | The 1.6–2.1× and 130 tok/s figures come from long text generations (5090 Q6_K: 61.9 → 130 tok/s at n=2, https://github.com/sudoingX/qwen38-mtp). The only vision+MTP figure is ~92 tok/s at n-max 5 (https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/14). The techniques report estimates 20–25% less model time. Est. for video 5: model time 1004 → ~700–800 s, wall ~830–930 s. | S | The GGUF may lack MTP tensors. Adds ~1.8–2.5 GB VRAM. Prefill hit: −3% on one CUDA GPU, though one 3090+3060 report halved it. MTP with the grammar-constrained tool output at temperature 0 is unverified. Needs n_parallel = 1, so it excludes leaf concurrency. Changes the fingerprint. | (1) `llama-server --help \| grep draft-mtp`, and check that the startup log shows the MTP/nextn layer. (2) Run MTP-off twice on video 5 to measure baseline nondeterminism of field strings. (3) Rerun with MTP. From per-call timings compare `predicted_per_second`, `prompt_per_second`, draft acceptance and labelled fields. Byte-diff field strings with MTP on and off; judge the differences against the MTP-off/MTP-off baseline. | U until `--help` and the startup log are checked on the server; † for the gain on our JSON + image turns |
| C3 | **The field-agent zoom loop takes ~4.6 requests and ~21–23 s per view** (115 calls / 25 views). The model starts from a 1024-px overview at ~27% of native resolution (`image_agent.py:140-146`, `:34-35`). Every zoom round re-sends all earlier images. | Send the upright sheet in the first message as overlapping full-width bands (≥25% overlap, Lanczos-sized to 32-px multiples).<br>Two bands at native width cannot cover the frame: a 2160-px-wide band at ≤4,096 tokens is at most ~1,940 px tall, and two of them with ≥25% overlap cover ~3,400 of 3,840 px. The sheet touches the frame edge in nearly every frame. So use either 3 native bands, or 2 bands at ~0.92× native (the reading report's 2160×2240 → 1984×2080 example), and record which one is used.<br>Set `MAX_MODEL_REQUESTS=2` (one zoom round) and `tool_choice='required'`. Bands need a new panel kind with its own counter and correct provenance, not `_Source.panel(context=False)`. | Recompute prefill first: about 8–12k new image tokens per view, roughly +3.5–5 s at 2,340 tok/s. That fits inside the ~10–16 s per view estimate (from 21–23 s); retest the estimate on the chosen band layout. If it holds: −175 to −325 s on video 5, ~54–64 s/sheet if applied alone (est.). | M | Precision is uncertain. Role and address-block handling probably improve; single-glyph reading may get worse, and the one zoom round is the mitigation. Overlap must exceed the tallest block, or both frames could cut it at the same place. Evidence boxes become coarser. | Branch from phase1-speed and run the A/B on video 5. Targets: field_agent calls per view ≤1.3; labelled ≥15/22; two-view agreement rate from `results.json` `image_recognition.views`. Cheap probe beforehand: `tools/read_zoom_images.py` on the 6 images in `/srv/sever/validation/accuracy-speed-v8-20261011/input-images`. | V (mechanism); band coverage arithmetic per critic |
| C4 | **The measurement cannot resolve ±5-point changes and is biased.** (a) All 4 consignee_address labels are postal-only, but field (4) prints organisation + ИНН/КПП + РЦ + address (`analysis/sample-05/selected-frames/upd-parties-crop.png`; `README.md:11-16`; `image_reader.py:59-63,177`). (b) The strict primary-timestamp matcher (`evaluation.py:108`) cost 4 fields on page-005. (c) There are no CIs or significance tests (`compare_runs.py`, `deploy/evaluate_server_results.py`). (d) Speed is not split by leaf class. (e) The 44 `not_visible_on_this_leaf` rows are skipped (`evaluation.py:128`), so speed changes on continuation leaves have no precision gate. | (1) Relabel the 4 values to the full block, keeping `legacy_value` and `field_boundary_rule`.<br>(2) Add bracket-tolerant matching as an extra metric: brackets for videos 4 and 5 only, global one-to-one DP in time order, human check of each bracket match.<br>(3) Score the not_visible rows as abstention checks (88–96 scored checks in total, no new labelling).<br>(4) Add Wilson CIs, exact McNemar and a cluster bootstrap.<br>(5) Report seconds per leaf class and per repeat showing.<br>(6) Fix `normalize`: quoted-day dates, two-digit years and «№» spacing fall back to the literal (`normalize.py:14-29`; `evaluation.py:49-53`). | Video 5 main: 16/22 → ~18/22 from the relabel. Up to 22/22 with brackets, assuming the page-005 values match the labels (team Q8). V7 35/44 → up to 43/44 (+8, ~18 points) as an upper bound; if V7 ran under the pre-11.10 address prompt, score its consignee_address against `legacy_value` instead. | S | Labels are provisional, so the relabel needs customer sign-off. Keep the strict metric as the headline (README.md:253). A bracket match can credit a neighbour shown mid-page-turn. | Unit tests from the reports, e.g. `wilson(35,44)=(0.655,0.888)` and `mcnemar_exact(6,0)=0.03125`. An abstention unit test: a not_visible row with a predicted value scores as wrong. Rescore today's two video-5 runs and V7 offline. | V (relabel, stats); U (extent of the bracket artifact beyond page-005) |

### Moderate

| # | Bottleneck (measured now) | Change | Estimated improvement | Effort | Risk | Test | Status |
|---|---|---|---|---|---|---|---|
| M1 | `prepare_zoom` drops the zoom tools after 8 crops or at request 7 (`image_agent.py:256-265`; `image_conflicts.py:461-465`). pydantic-ai lists tools before the system text (`pydantic_ai/models/__init__.py:264-265`), so the prompt changes ~10–20 tokens in and the whole request is re-processed. | Keep the tool list constant. Rely on the existing decline message (`image_agent.py:278-289`). Hide the tools only on the final allowed request. | Avoids re-processing ~10–17k tokens (4–7 s) per affected request. 1–5% of model time; frequency unknown. | S | One extra short request if the model zooms after the limit. Changes the prompt hash. | In the phase1 trace, find requests after a `zoom_08` proof or at page_request ≥7. Compare their `cache_n` with the previous total. | U (Qwen3.8 template order and frequency unverified) |
| M2 | Possible full re-prefill of the ~6.7k-token shared prefix at each leaf's first view. The model is hybrid DeltaNet. b7822 disables both context checkpoints and the RAM prompt cache when an mmproj is loaded, so checkpoint flags help only if v0.6.0 enables them with an mmproj. | Make this conditional on a P0 startup-log check:<br>(a) If v0.6.0 enables checkpoints with an mmproj and `cache_n` at run start ≈ 0, try `--checkpoint-min-step 1024 --ctx-checkpoints 64`.<br>(b) If not, drop the flag tuning and use serving finding 9's fallback: shrink the static prefix (drop the "Omit" fields, deduplicate exact rule repeats; target ~5k tokens). | (a) 0 to ~70 s per video. The 25 × 6.7k upper bound probably overstates it, since 68% of input is already cached. (b) ~25 s per video (6.7k → 5k). | S | (a) More host RAM per checkpoint (~150 MiB each); flag names may differ in v0.6.0. (b) The rule repetitions were added to fix observed errors; drop only exact duplicates and A/B. | Startup log: checkpoint/prompt-cache lines with the mmproj loaded. Trace: `cache_n`/`prompt_n` of every run's first request. One `-lv 4` run, then grep for "forcing full prompt re-processing". | U |
| M3 | Output tokens dominate (222–236 per call). The schema has fields marked "Omit" (`image_reader.py:163,211`), long `zoom_NN_rN` ids (`image_agent.py:149`), `tool_choice: auto` (`:329`), and joint-review `evidence_text` that repeats values (`image_conflicts.py:163`, `resolution_reason` max 2000). | Measure per-kind and per-tool output tokens from the trace first. Then:<br>- drop the "Omit" fields (requires `_validate_reading` changes at `image_reader.py:397-417`);<br>- use short panel ids;<br>- set `tool_choice='required'`;<br>- cap the heading to title + ОКУД;<br>- make joint-review evidence caption-only with a short reason.<br>Keep `observed_form`, which `report.py:312` and `native_pipeline.py:902` use. | Est. −60 to −100 s per video for submit_reading. Joint-review compaction applies to ~9 final submits, not 39 calls. | S | Reverses some prompt-tuning history. The schema hash changes. | Trace sums of `predicted_n` per kind and per tool name, then an A/B on video 5. | U |
| M4 | The per-frame schema allows status "conflict" (`schema.py:7`; `image_reader.py:162`). One such answer makes fusion mark the field conflict (`image_fusion.py:181-185`). Only a joint review can clear it, and any later per-frame conflict reopens it (`native_pipeline.py:536-547`). | Give the per-frame `ImageField` its own `Literal['recognized','not_present','unreadable']`. Fusion and joint review keep "conflict". | 0 if the model never emits it; otherwise ~30–60 s per affected leaf. | S | Joint review shares the field models (`image_conflicts.py:24-25,33`). | Count per-view conflict statuses in the phase1, main and V7 `results.json`. | V |
| M5 | Image sizing.<br>- The 576×1024 overview is upscaled by the server to 768×1376: 1,032 tokens with no added detail.<br>- Small crops are upscaled to the 1,024-token floor. This contradicts `image_agent.py:77` and `README.md:213` (`--image-min-tokens 1024`, service file line 12; b7822 `clip.cpp:2357-2361`, `:2834-2843`). | Build the overview at 768×1376 directly from the native frame, optionally cropped to the sheet box. It stays at 1,032 tokens client-side.<br>Then A/B `--image-min-tokens 256`. The flag is server-wide, but the overview and triage images are already above the floor, so the change reaches only crops and the 144-token orientation images.<br>The `clip.cpp:1167-1171` grounding warning concerns images the model must localise within. Here the model chooses crop coordinates on the whole upright frame via the overview (`image_agent.py:75`). Crops are only transcribed, and orientation images only rotated.<br>Update the instruction text to match. | Free 1.33× real detail in the overview. With min-tokens 256: ~−790 tokens per small crop, est. −50 to −70 s per video. | S | Thin single-line crops may read worse without the server upscale. Changes the fingerprint and batch. | (1) Triage `prompt_n` (~1.37k if upscaled, ~0.9k if not) confirms the resize rule on v0.6.0. (2) A/B on the 4 labelled videos. Pass: crop-cited fields no worse on the 44 labels, and orientation 15/15 correct. | V (math on b7822); † on v0.6.0 |
| M6 | After a non-terminal review the no-progress window stays at 3 views. Any new spelling of a field already in conflict counts as progress (`native_pipeline.py:43-51, 456-458`). Conflict is permanent (`image_fusion.py:183`). | After an unresolved fields review, use a window of 1. Count progress only on unreadable or pending fields and the document type. | page-0015: −2 views, ~−10 calls, ~−90 s. Overlaps C1, so the savings do not add. | S | Update `tests/test_native_conflict_pipeline.py:357-372`. | Offline prefix fusion on the V7 batch (analysis only), then an A/B. Pass: page-0015 ≤24 requests; no false values or roles on the not_visible rows. | V |
| M7 | `--parallel 1`. Leaf concurrency exists (`agents.py:133`, `cli.py:29-30`) but is off. | `--parallel 2 --kv-unified --ctx-size 131072` with `HARNESS_LEAF_CONCURRENCY=2`, pinning each worker to `id_slot` via `extra_body`. | −27 to −30% s/sheet on video 5 at N=2 (est., simulation: 75.6 → 53–55 s). Up to ~1.45× batch throughput at N=2. Capped by page-0015 (301 s). | S/M | Incompatible with MTP today. `--ctx-size 131072` adds ~2 GB of q8_0 KV against ~5.9 GiB headroom. Without `--kv-unified`, an overflow returns HTTP 400, which is fatal and stops the batch (`image_agent.py:348,380-383`). Not batch-invariant. `HARNESS_MODEL_TIMEOUT=600` may be hit by a joint review queued behind another slot's prefill. | `ab_speed_test.sh` with `REF_A=REF_B=phase1-speed` and `LEAF_CONCURRENCY_B=2`. Check peak VRAM in nvidia-smi. Grep the log for slot steals, "Context size has been exceeded" and request timeouts. | V (b7822 code); A/B needed against C2 |
| M8 | Primary and seed frame choice.<br>- 26/51 primaries sit at the first or last sample of a leaf (19.5 expected).<br>- 39/54 first seeds are the adjacent 0.53 s sample (`video.py:478-487, 496-503`).<br>- `frame_pool.py:133-136` computes novelty but never uses it.<br>- Phase 1 reads exactly 2 views on 14/15 leaves. | Add a centrality weight and full-timeline motion to the primary score. Choose the seed ≥1.0 s away and use edge agreement as a yes/no gate. Also change the read order (`native_pipeline.py:429-432`). Bump `algorithm_version` (`video.py:414`). | Likely recovers page-005 under strict matching (+4 fields on the headline metric). Fewer correlated misreads (unmeasured). | S | Short leaves have no inner sample, so keep current behaviour there. Decorrelated frames may trigger more joint reviews. The statistics come from interim algorithm_version 3. | CPU only: `analyze_video` A vs B on the 4 videos plus `evaluate()`. Then a model A/B. | V |
| M9 | The joint-review «он же» rule (`image_conflicts.py:61-63`) contradicts the single-frame contract (`image_reader.py:138-140`). Joint values overwrite fused ones (`native_pipeline.py:692`). This produced phase 1's only regression. | Add to the joint instructions: raw_value never contains «он же». The reference line belongs in evidence_text. | +0/+1 field on current labels; prevents the same prefix across the batch. | S | The wording must still require the entire printed block. | Scan joint-resolved fields for "он же". A/B on page-0004. | V |
| M10 | Silent glyph errors.<br>- Fusion uses only NFKC + casefold (`image_fusion.py:39-41`), so Cyrillic/Latin homoglyphs pass.<br>- Embedded ИНН are never checksummed (`native_pipeline.py:1045-1046`), although `sever/inn.py:39` exists. | Add a one-line script hint to the instructions. Flag (never repair) tokens that mix Latin and Cyrillic letters. Use `validate_inn` as a confirmation veto. | Small but certain precision gain. MDPBench names Cyrillic/Latin lookalikes as the Russian failure mode (https://arxiv.org/html/2603.28130v1). | S | Exempt Latin-only tokens. Parse the "ИНН/КПП d/d" layout. The labelled set has 1 ИНН, so the gain cannot be measured on current labels. | Offline scan of V7 and phase1 outputs. | V |
| M11 | TTN commodity continuations (pages 012/014) are typed ТОРГ-12 in both arms. That is 2 of the 3 video-5 type errors. It possibly also explains video 4's 008/010 and 4 of 6 V7 target-type errors (†; the segmentation report says "probably", unverified). | No verified fix yet. Diagnose by checking per-view `heading_text`/`observed_form` on these leaves, then consider prompt guidance on continuation pages. | Up to +2 target types on video 5. | S–M | — | Dump per-view `heading_text`/`observed_form` on pages 012/014 from both arms. After any prompt change, pass = type ТТН on both views of 012 and 014, and no new ТОРГ-12 → ТТН flips on sample-04. | Flagged (no finding) |
| M12 | Role confusion: the document creator is read as consignor, a known error class. M9 covers only «он же»; no item addresses the general case, and the frequency is unmeasured. | No verified fix yet. Measure first: count per-view consignor/consignee values that equal the creator (seller) block, across V7 and phase1. Consider prompt or schema guidance only after the count. | Unknown. | S (measure) | The not_visible role rows are a subset of the 44: 16 of the 44 are TTN commodity continuations with no role lines. | (1) Abstention-check pass rate on the not_visible role rows (UPD continuations; on video 5, pages 013/015). (2) Per-view count of consignor values equal to the creator block. Pass after any change: the count falls and there are no new false values on the abstention checks. | Flagged (no finding) |
| M13 | The 2000-call job budget was exhausted on 20260818-3 under V7: it failed at 54/58 pages, averaging ~34 calls/page. Whether phase1's no-progress stop prevents this is unmeasured. | Rerun phase1 on 20260818-3. Consider a per-leaf backstop only if it still exhausts the budget; a hard cap needs owner or customer agreement (README:38). | Possibly the 4 lost pages of video 3; unmeasured. | S | Video 3 may be a breadth problem (~34 calls/page), not a per-leaf tail. | Rerun phase1 on 20260818-3. Pass: ≤ ~40 calls/leaf, no budget stop, 58/58 pages. | U |

### Minor

| # | Item (measured) | Change | Estimate | Effort | Test / pass threshold | Status |
|---|---|---|---|---|---|---|
| m1 | Orientation: 15 calls × ~2.1–2.3 s, four 1,032-token images each (`image_orientation.py:75-81`). | Tile the 4 rotations into one ~1088×1088 image, or answer rotation in the first triage call. | ~−20 to −35 s per video | S | Orientation 15/15 correct against the phase1 reference; orientation seconds ≤ half of current. | V |
| m2 | Conflict-only leaves keep reading (`native_pipeline.py:919-922`). The final-review frame chooser is all-or-nothing (`:226-254, 626`). | Read 1 fresh view, then run the final review. Choose frames that maximise fully covered fields. | 0 s on video 5; tail guard | S/M | Unit test: two conflict fields whose union needs 5 frames while each needs ≤4. Offline count of final-phase `blocked_fields` in V7. Gate: no false values or roles on the not_visible rows. | V (code); unmeasured |
| m3 | Over-broad blockers on the other-side stop (`native_pipeline.py:810, 813, 823, 827-829`). | Relax the field-unrelated errors and the failed budget views. Keep persistence errors blocking. | 0 on video 5; hypothetical | S | Offline over V7: classify which guard blocked each eligible leaf; one unit test per relaxed guard. Gate: no false values or roles on the not_visible rows. | U |
| m4 | `max_tokens` is 8192 for agent and joint calls (`local-provider.env.example:7`). A runaway answer costs ~134 s. | Cap per kind at ~2× the observed max. | Tail only | S | Histogram of `predicted_n` per kind from the trace; after the cap, zero `finish_reason=length` and no field changes on video 5. | V |
| m5 | Fusion compares dates and numbers literally (`image_fusion.py:53-60`), which is stricter than the evaluator. | Measure formatting-only conflicts first. | Unknown | S | Offline: count conflict fields whose `value_groups` compare equal under `normalize.comparison()`. Act only if the count is >0. | V |
| m6 | 2-sample leaves skip short-candidate confirmation (`video.py:291`). The worst direction alone decides "partial page" (`:141-144`). | Confirm and merge weak fragments, extending `video.py:342-349`. | Only where fragments occur; current count unmeasured | M | Unit tests: a skin blob over 30% stays same_page; a new printed sheet over 30% stays partial. Offline `analyze_video` + `evaluate`: unmatched predictions fall, and target detection stays 20/20 (bracket). | V (code) |
| m7 | `frame_pool` ranks once on 384-px whole-sheet sharpness (`frame_pool.py:69,128-138`). | Per-region sharpness for unresolved fields. Glare is ~0 in this footage. | Hard leaves only; the pool ran on 1/15 phase1 leaves | M | Retrospective ROC AUC of the new score on main's hard leaves 0012–0015; then an A/B on page-0015 seconds and views. | V (mechanism) |
| m8 | Two full HEVC decodes (`video.py:443, 531-533`). | Seeked or streamed original extraction. | 2–4% | M | Video-analysis seconds ≤ −2%, with identical leaves. | U |
| m9 | Default `-ub 512`. | Sweep 1024/2048 with `llama-bench`. | Est. −3 to −5% at N=1 | S | `llama-bench` pp at `-ub` 1024/2048 ≥ +10%, with VRAM ≤29.5 GB. | U |
| m10 | q8_0 KV (KL ~0.024). | f16 KV, or keep q8_0 if MTP needs the VRAM. | Not a speed lever | S | `llama-perplexity --kl-divergence` q8_0 vs f16 on a recorded multimodal prompt text. Video-5 labelled fields unchanged; peak VRAM within budget. | † |

**Do not do** (verified or reasoned against):
- Perspective warp. 22/22 and 197/198 sampled sheets touch the frame edge, and `_inverse_affine` is affine-only (`image_reader.py:262-266`). A sheet-box crop is enough.
- 0.25 s base sampling. The effective step is already 0.533 s, and no sampling loss was found.
- Switching the grammar engine. 61 tok/s is about 77% of the bandwidth ceiling.
- Draft-model speculation, `--cache-reuse`, or the RAM prompt cache with an mmproj loaded. All are disabled in b7822. MTP (C2) is the exception only if v0.6.0 lifts the mmproj restriction; check that first.
- Lowering `--image-min-tokens` below the overview's 1,032 tokens. The grounding warning is at `clip.cpp:1167-1171`. M5 lowers the floor only for crops and orientation images.
- Majority voting (README:205; `tests/test_image_fusion.py:235-243`).
- A hard per-leaf cap (README:38) without owner or customer agreement.
- A per-form split schema.
- Reusing frame 1's model-chosen rectangles as frame k's crops without customer sign-off. They are part of a stored model answer.

## 3. Optimization roadmap

| P | Change | Speed effect | Precision effect | Effort | Metric affected | Depends on labels/benchmark | Test / pass threshold |
|---|---|---|---|---|---|---|---|
| P0 | Offline trace and startup-log analysis, no model calls.<br>- From the phase1 trace: per-kind p50/p95 timings, cache continuity, first-request `cache_n`, peak context, output tokens per tool, per-view conflict count.<br>- From startup: architecture, KV/recurrent lines, `image_min/max_pixels`, MTP tensors, checkpoint/prompt-cache behaviour with the mmproj, `--help` flags. | Decides M1–M3, M5, M7 and the M2 branch | — | S | all speed metrics | No | A named output file, e.g. `analysis/ab-speed-20261017/trace-breakdown.json`, with per-kind p50/p95 of `prompt_n`, `cache_n`, `predicted_n`, `prompt_ms`, `predicted_ms`; first-request `cache_n`; peak context per kind; output tokens per tool; and the startup-log grep results. |
| P0 | Phase1 baseline on the full 11-video batch | — (baseline) | — | M | s/sheet batch, calls/sheet | No | Per-video and batch s/sheet, calls/sheet and budget stops, reported against V7 (174 s, ~34 calls). |
| P0 | C4 metric fixes:<br>- relabel consignee_address with `legacy_value`;<br>- bracket metric;<br>- abstention checks on the 44 not_visible rows;<br>- Wilson/McNemar;<br>- per-class speed;<br>- normalize fallbacks;<br>- remove the 4-video hard check (`deploy/evaluate_server_results.py:140-141`);<br>- CSV importer. | — | Removes up to ~18 pts of downward bias on V7 (upper bound) | S | labelled fields, detection, abstention | Customer approval for the relabel | Unit tests from C4; today's video-5 runs rescored with CIs, p-values and abstention counts. |
| P0 | C2 MTP at `--parallel 1` (only if `--help` and the startup log confirm support) | −20 to −30% model time (1004 → ~700–800 s, est.) | Neutral; verify byte diffs against an MTP-off/MTP-off baseline | S | s/call, s/sheet | Labels as a regression gate only | §5 thresholds |
| P0 | M4 per-frame conflict removal + M9 «он же» rule | Small | + (page-0004, locked conflicts) | S | labelled fields, joint calls | 22 labels (gate) | Per-view conflict count → 0; no «он же» in joint raw_value; labelled ≥15/22. |
| P0 | C1 two-frame other-side stop | −200 to −340 s on video 5 (est.) | Expected neutral; gated by the abstention checks | M | calls/sheet, s/sheet | Abstention checks | C1 targets plus no false values or roles on the not_visible rows. |
| P1 | Model A/B (section 4): Qwen3.6-35B-A3B, Qwen3.6-27B, Qwen3.8-27B at a cheaper quant, vs a Qwen3.8-27B Q6 control | MoE: −32% (2 clips, pre-v6 OCR-gated pipeline, accepted-field counts, not accuracy) to −60% (est.). Cheaper quant: ~1.2–1.4× decode (est.). | MWS suggests + on OCR/KIE, − on grounding | M | all | Yes: non-inferiority on 44 labels at minimum; better after the first labelling round | §4 decision rule |
| P1 | M5 image sizing; M1 constant tool list; M2 per the P0 branch | −5 to −15% | Small + | S | s/call | 44 labels gate | M5, M1, M2 tests |
| P1 | M3 output compaction (after the trace) | −5 to −9% | Neutral | S | s/call | Gate | `predicted_n` per submit_reading down; labelled ≥15/22. |
| P1 | C3 band-based one-request reading | −15 to −29% wall (est., retest after the band recompute) | Uncertain (±) | M | calls/view, labelled fields | Yes: needs ≥44 labels plus two-view agreement | C3 targets |
| P1 | M8 primary centrality + seed independence + read order; segmentation scorer and sweep harness (CPU) | Neutral | +4 fields strict (page-005); fewer correlated errors | S/M | detection, labelled fields | Truth brackets (videos 4–5) | Strict target misses 1 → 0 on video 5; first/last-sample primaries ≤40%. |
| P1 | M6 no-progress window; M10 homoglyph flag + INN veto; M12/M13 measurements | −90 s on 0015 (overlaps C1) | Small + | S | calls/leaf, precision | Gate | M6, M10, M12, M13 tests |
| P1 | Labelling round 1 (section 6) | — | Makes ~6-pt changes visible | M | all accuracy | — | Wilson half-width ≤0.07; double-labelled agreement ≥0.95. |
| P2 | M7 `--parallel 2 --kv-unified` + `id_slot` (A/B against MTP) | −27 to −30% s/sheet at N=2 (est.); ~1.45× batch throughput | Batch noise | S/M | s/sheet, throughput | Gate | M7 test; peak VRAM ≤ ~31 GB; zero context-exceeded errors and timeouts. |
| P2 | m1 orientation; m2/m3 tail guards; m4 caps | −2 to −4%; tail | Neutral / small + | S | s/sheet, job failures | No (abstention gate for m2/m3) | Minor-table tests |
| P2 | m6 fragment pruning; repeat-showing link (metadata only) | Only where fragments or repeats occur (repeats = 56% of phase1 time on video 5) | Customer-facing duplicates | M | leaves per interval | Customer decision on repeats | Offline precision/recall of the link rule against truth `repeat_of` on the 4 videos; fragments inside target intervals fall; target detection stays 20/20 (bracket). |
| P2 | Offline OCR-specialist bake-off as veto/hint candidates only: PaddleOCR-VL-1.5/1.6, dots.ocr/mocr, HunyuanOCR-1.5, PP-OCRv5 eslav, and the existing Tesseract path | Unknown | Possible + on glyph errors | S offline / L to integrate | literal hit rate on 44 fields | Yes | Literal hit rate ≥ the VLM's on the 44 fields (on ≥2 distinct frames). Risk: this conflicts with the README design statement of no separate classifier or prior Tesseract (README.md:38), and the `sever/ocr.py` OCR cache conflicts with the no-replay rule, so OCR must run fresh, offline only. |
| P2 | vLLM 0.31 + nvidia/Qwen3.8-27B-NVFP4 (+MTP), only if ≥4-stream throughput becomes the goal | 2.5–3× video wall at 4 streams (est.) | W4A4 risk, no OCR evaluation | L | throughput | Full 44+ label gate | Replay video-5 at concurrency 1 and 4. Pass: ≥2.5× aggregate throughput at 4 streams, labelled fields non-inferior to llama.cpp Q6, prefix-cache hit rate reported (current 68%). |

## 4. Model choice

Weights:
- Russian OCR evidence 35%
- Decode speed class 20%
- Fits 32 GB at a usable quant with ≥16k context 15%
- llama.cpp/vLLM support maturity 15%
- Structured/tool output 10%
- License 5%

Scores run 1–5. † marks an unverified cell. No candidate card was opened by a verify pass, so every row is "existence/licence unchecked †" until someone opens it.

Main Russian evidence: MWS Vision Bench, a private test set of Russian business documents (https://github.com/mts-ai/MWS-Vision-Bench). Columns are img→text / KIE / grounding. Inference settings are unpublished. On its 119-item KIE subset the split noise is ~0.06.

| Model | URL (card unchecked †) | Russian OCR evidence | Fits 32 GB | Decode speed | Serving maturity | Structured / tool | License † | Weighted |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.8-27B Q6 (current)** | https://huggingface.co/Qwen/Qwen3.8-27B | 3: 0.797 / 0.807 / **0.223** (best Qwen grounding); OCRBench v2 EN 63.5 | 4: 26.2–26.7 GB measured | 2: 61 tok/s measured; MTP gain on vision turns † | 5: in production | 5: in production | 5: Apache-2.0 † | **3.55** |
| **Qwen3.6-27B** (Q6_K 22.5 GB) | https://huggingface.co/Qwen/Qwen3.6-27B | 4: **0.861 / 0.893** / 0.062; validation KIE only 0.833; OCRBench v2 EN 68.3 | 4: ~25.5–26.5 GB † | 2: same architecture; MTP GGUF exists (https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF) | 4: unsloth GGUF, own mmproj | 4: qwen3_coder XML; check submit_reading parsing †; template defaults to thinking | 5 † | **3.65** |
| **Qwen3.6-35B-A3B** (MoE, 3B active) | https://huggingface.co/Qwen/Qwen3.6-35B-A3B | 4: 0.847 / 0.840 / 0.094; OCRBench v2 EN 65.5. In-house: same 9/10 *accepted* fields on 2 clips with the pre-v6 OCR-gated pipeline (accepted-field counts, not accuracy; `deploy/README.md:236-246`). | 3: Q6 already ran on this server (`deploy/README.md:260`), leaving no room for `--parallel`; UD-Q5_K_M 26.5 GB | 5: ~180 tok/s tg on 5090 (llama-bench, empty context; https://zenn.dev/toki_mwc/articles/rtx5090-qwen36-35b-a3b-llmacpp-bench). 4.2× the 27B at Q8 on RTX PRO 6000 (https://jarvislabs.ai/blog/qwen36-mtp-llamacpp-rtxpro6000). MTP adds only 1.17×. | 4: mainline; the unsloth MTP repo says MTP + `--mmproj` is not yet supported, so use the non-MTP GGUF † | 4 † | 5 † | **4.10** |
| **Gemma 4 31B-it** | https://huggingface.co/google/gemma-4-31B-it | 2: 0.734 / 0.793 / 0.043 | 3: Q5_K_M 21.7 + 1.2 GB; SWA cache cost † | 2: dense 31B † | 4: official ggml-org GGUF | 3: native tools, different template; recommended temperature 1.0 | 5 † | **2.70** |
| **Mistral Small 3.2 24B** | https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506 | 1.5: overall 0.577, OCR 0.748, KIE 0.664 | 4 † (~19–20 GB at Q6) | 3 † | 3: only 3.1 is listed in the llama.cpp multimodal docs | 3: vLLM mistral parser; llama.cpp † | 5 † | **2.73** |
| **Qwen3.5-9B** (helper only) | https://huggingface.co/Qwen/Qwen3.5-9B | 2.5: MWS 0.772 / 0.789; MDPBench RU figures † (reports disagree on the column) | 5 | 4 † | 4 | 4 | 5 † | 3.68 (not eligible as the field reader) |

**Deal-breakers:**
- **Too large for 32 GB:** GLM-4.6V 106B; Llama 4 Scout 109B (Russian also not listed); Mistral Small 4 119B; Qwen3.8 2.4T; GLM-5.3-Flash. GLM-5.3-Flash is a 320B MoE (russian-ocr report), despite grounding of ~0.6 on MWS.
- **FP8 27B:** ~29 GB, so it does not fit alongside vision and KV.
- **GLM-4.6V-Flash:** zh/en only.
- **InternVL3.5:** no Russian OCR evidence. InternVL2-76B scored 13.18 on CC-OCR Russian.
- **Cotype Light 3:** strong MWS KIE (0.873 val), but no open weights were found, and it is scored on its vendor's own benchmark.
- **Gemma 4:** a hard image cap of 1,120 tokens, against our 4,096. Not fatal, but it puts all the load on crops.
- **Qwen3.5-9B as the field reader:** MWS KIE 0.789, against 0.807–0.893 for the 27B-class models.
- **Specialist OCR models (PaddleOCR-VL, dots.ocr, HunyuanOCR):** no key-information extraction or tool calling, so they are usable only as a veto or hint.

**Omitted, not scored:** Gemma 4 26B-A4B, a MoE with OmniDocBench edit 0.149, raised by the vlm-general report as a faster non-Qwen alternative. Mistral Small 3.2 also qualifies as a non-Qwen control.

**Two caveats.**
- Qwen's own card puts Qwen3.8-27B ahead of Qwen3.6-27B on every shared benchmark (e.g. OmniDocBench 1.5 91.1 vs 89.4). Qwen3.6 leads only on a single third-party Russian benchmark.
- The user explicitly chose Qwen3.8-27B (`deploy/README.md:166,260`), so any swap is the user's decision.

**Benchmark first (in order):**
1. **Qwen3.6-35B-A3B.** Q6 is proven to fit; also try UD-Q5_K_M. This is the decode-speed lever, with Russian OCR/KIE at or above the current model.
2. **Qwen3.6-27B Q6_K.** The accuracy-first drop-in, same architecture.
3. **Qwen3.8-27B at Q5_K/Q4_K_M.** A cheaper quant of the same model as a decode-speed arm (~1.2–1.4× decode, est., techniques report). It needs the same accuracy gate.
4. **Qwen3.8-27B Q6** as the control. Gemma 4 31B or Mistral Small 3.2 only as a later non-Qwen control.

**Exactly how.** `deploy/ab_speed_test.sh` compares two git refs against one fixed model server, and it never changes the model service. A model A/B is therefore an operator procedure:

1. Put the server in the idle state (`deploy/README.md:627-662`).
   - Point `/etc/sever-tsl/local-model.env` `MODEL_PATH`/`MMPROJ_PATH`/`MODEL_ALIAS` at the candidate, with its matching mmproj. The Qwen3.6 mmproj is not interchangeable.
   - Keep every other flag identical: ctx 65536, q8_0 KV, image tokens 1024/4096, thinking off, temperature 0.
   - For Qwen3.6, the template defaults to thinking. Verify that `enable_thinking:false` takes effect: no reasoning content, and output tokens in line with the control.
   - Restart sever-model and record the GGUF/mmproj hashes and the `image_min/max_pixels` startup line.
2. Run the same ref (phase1-speed HEAD) on `/srv/sever/input/20260818-5_132107.mp4` first, then on the other 3 labelled videos.
   - Run once per model, under the shared inference lock, with no overlapping jobs.
   - Run each arm twice on video 5 to measure run-to-run variance.
3. Compare with `tools/compare_runs.py <runA> <runB> --truth analysis/ground-truth-expanded-provisional.json`, extended with per-kind timings from P0. Report:
   - labelled fields (after the relabel), with McNemar
   - abstention checks on the not_visible rows
   - target types
   - conflict and unreadable counts
   - joint_review calls
   - zoom crops per view, and the share of crops later cited as evidence (a grounding proxy)
   - `predicted_per_second` and `prompt_per_second`, calls per leaf, peak VRAM
4. Decision rule. "Not significantly worse" on 44 labels is near-automatic, because a regression needs ≥6–0 flips to reach significance. Use a non-inferiority margin instead:
   - **MoE or cheaper quant:** adopt if it loses ≤1 net field on the 44 labels, adds no new false values on the abstention checks, and cuts wall time by ≥30%.
   - **Qwen3.6-27B:** adopt only if accuracy gains are seen and crop misses do not rise. Claims of improvement wait for the ~150-label round.

## 5. Serving-engine recommendation

Stay on llama.cpp. On this workload (one GPU, mostly sequential calls, 68% prefix reuse, grammar-constrained tool calls) it runs at about 77% of the batch-1 bandwidth ceiling. vLLM's advantages are concurrency (936 vs 369 tok/s at 32 streams on a 5090) and prefill. Its costs:
- Prefix caching on hybrid models is experimental.
- It drops to 0% cache hits with MTP unless the state is bf16 and blocks are aligned.
- The 27B fits only as NVFP4 W4A4, which has no OCR evaluation.

TensorRT-LLM has no KV block reuse for this architecture, and SGLang's DFlash2 does not support greedy decoding.

**The single most promising change:** MTP self-speculative decoding on the existing model and binary. Add `--spec-type draft-mtp --spec-draft-n-max 2` to `deploy/sever-model.service:12` (sweep 1–3) and keep `--parallel 1`.

This is conditional. The serving report verified that b7822 refuses speculative decoding when an mmproj is loaded, and it advised against spending time on speculation, but that advice was written against b7822. The change works only if the deployed v0.6.0 is newer and contains #28715.

Supporting evidence:
- https://github.com/ggml-org/llama.cpp/pull/22673: merged 2026-05-16, "MTP is compatible with Vision input".
- v0.4.1 release notes: "Fixed speculative decoding after image input (#28715)".
- https://github.com/sudoingX/qwen38-mtp: 5090, Q6_K, 61.9 → 130 tok/s. This is a long text generation.
- https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/14: vision+MTP ~92 tok/s at n-max 5, the only vision figure.

Preconditions:
- `llama-server --help` lists `draft-mtp`.
- The startup log loads an MTP/nextn layer. The deployed UD-Q6 GGUF may lack it.
- The v0.6.0 build contains #28715.
- MTP works with the grammar-constrained tool output at temperature 0. This is unverified, so check it with the diff test below.
- Peak VRAM ≤ ~29.5 GB.

**Measurement that confirms it:**
1. Run video 5 phase1-speed twice with MTP off, to measure baseline nondeterminism of field strings.
2. Rerun with MTP on.

From the per-call timing log, the change is confirmed if all of these hold:
- median `predicted_per_second` ≥ 90 (from 61)
- `prompt_per_second` ≥ ~2,200 (from 2,340)
- draft acceptance ≥ 0.5
- total model time ≤ 800 s (from 1004 s; est. 700–800 s)
- labelled fields unchanged, and no new false values on the abstention checks
- MTP-on vs MTP-off field-string differences no larger than the MTP-off/MTP-off differences, or each one individually explained

Then sweep n-max 1/2/3. If prefill halves, as one report saw on a 3090+3060, roll back.

## 6. Labelling plan

**Goal.** The headline accuracy needs ~250 independent value labels for a ±5-point 95% half-width at p≈0.8. That becomes 370–490 rows once repeat and within-sheet clustering are accounted for. A paired A/B detecting a 5-point change at 80% power needs ~250–310 paired fields.

Today there are 44 rows:
- 21 of them are repeat showings, and they hold only 7 distinct values from 2 document sets.
- One glyph class (УМНЫЙ РИТЕЙЛ × 8) can swing the score by 18 points.
- TORG-12 and ТрН (9 of the 19 fields) have zero video labels.

**Tooling first** (S): remove the 4-video check (`deploy/evaluate_server_results.py:140-141`) and add `tools/import_review_csv.py`. Nothing currently reads back `corrected_value` or `review_status` (`tools/build_review_table.py:43-51,113`).

**Order:**
1. **15 min.** Migrate the 4 consignee_address labels to the entire field-(4) block, with `legacy_value`. Check the sample-04 crop separately. Confirm the 4 consignor_address labels («он же» → seller 2а, postal only) and decide the boundary convention.
2. **20 min.** Spot-check the 44 `not_visible_on_this_leaf` rows and set `review_status`; these become the abstention checks. Send customer question Q3 (UPD continuation number/date).
3. **~12 min.** Add the literal heading for each unknown leaf: 68 intervals, excluding blank reverse sides. Score an `unknown_with_heading` check against `heading_text` (`image_reader.py:213-214`).
4. **~25–30 min per video** for the five short unlabelled videos. These 70 leaves are 40% of V7 compute. Order:

   | Video | Leaves | Calls/leaf |
   |---|---|---|
   | 20260821_144838 | 11 | 72 |
   | 20260821_145110 | 14 | 56 |
   | 20260818-6_131851 | 17 | 49 |
   | 20260821_145342 | 13 | 46 |
   | 20260818_131702 | 15 | 34 |

   For each video:
   - Make contact sheets with `tools/inspect_video.py <video> analysis/<id> --interval 0.5`.
   - Group samples into leaves. Boundaries are midpoints, and brackets are the adjacent samples. Set `repeat_of`.
   - Type each leaf.
   - Extract full-resolution frames with `ffmpeg -ss`, independent of the pipeline's frame choice.
   - Mark each of the 19 fields as visible_readable (value + raw literal; addresses as the entire block), not_visible, semantic_pending or illegible.
   - Never look at model outputs.
5. Put the 20260821_* videos in partition `test`. Video 5 stays `development` as the A/B video. Also rescore sample-04 for every change, to guard against overfitting to video 5.
6. A second person independently re-reads ~10% of the new value rows (~20). Target agreement ≥0.95.

**Expected yield (est.):**
- ~45 target leaves, giving ~100–120 value labels. With the existing 44, that is ~150 value labels (±6.5 points).
- Reaching ±5 needs the target leaves of 20260818-3 (58 leaves) and 20260821_144256 (62 leaves), another ~2–3 h.
- Prioritise any TORG-12/ТрН fronts found.
- Use V7 types only to rank leaves for reading, never as values.

## 7. Open questions for the customer and team

**Customer**
1. Should address and party labels follow the 11.10 entire-block rule? If yes, approve the 4 consignee_address relabels. How should the «он же» consignor_address be labelled?
2. Is the unit of scoring a visible leaf or a physical document? May repeat showings be linked? If so, may a repeat's reading use the original showing's confirmed frames? Repeats are 56% of phase1 time on video 5.
3. Does a UPD continuation's own number and date count as invoice_number/date? Are fields printed only on the other side excluded or merged?
4. Is run-to-run non-bit-reproducibility (MTP verification, batched slots) acceptable under the acceptance protocol?
5. May a frame-1 model-chosen rectangle be reused as frame-k crop coordinates, or does that count as replaying a stored answer?
6. Is a strict literal_exact metric needed alongside the presentation-equivalent one? `normalize.py:41-52` folds legal-form expansions and case, which the literal rule forbids.

**Team**
1. Does `finish-evaluation.json` split V7's 8 misses as 6 non-target (video 4), 1 target (video 5 page-005) and 1 in video 2?
2. Is Qwen3.8-27B really hybrid with 32-px tokens on the deployed GGUF and mmproj? Does v0.6.0 keep b7822's mtmd rules: no speculative decoding, no RAM prompt cache, no checkpoints with mmproj, bilinear resize to the min/max window?
3. Does the deployed UD-Q6 GGUF contain MTP tensors? Does v0.6.0 contain #28715 and accept `draft-mtp` together with `--mmproj`? Does MTP leave grammar-constrained tool output unchanged at temperature 0?
4. Is "seconds per sheet" single-video latency (bounded by the hardest leaf, 301 s) or 11-video batch throughput? This decides MTP vs `--parallel N` vs the MoE.
5. Why did main's initial joint review fail to settle pages 0012–0014 when phase 1's did? Is phase 1's hard-leaf gain model nondeterminism, and therefore fragile?
6. How often does the tool list change mid-run (M1)? How often is per-view `cache_n` ≈ 0 at the start of a leaf (M2)? The P0 trace analysis answers both.
7. How many leaves does the current code (algorithm_version 4) produce on the 7 unlabelled videos, i.e. what is the real fragment rate?
8. Did the 4 values page-0005 confirmed today match the sample-05-page-005 labels?
9. Are there TORG-12 or ТрН fronts in the unlabelled videos?
10. The Qwen3.8 Flash entry on MWS (grounding 0.301 validation / 0.264 test): is it open-weight, and does it fit 32 GB? This has not been checked. GLM-5.3-Flash was checked (320B MoE, does not fit).
11. Does phase1 on 20260818-3 still exhaust the 2000-call budget (M13)?