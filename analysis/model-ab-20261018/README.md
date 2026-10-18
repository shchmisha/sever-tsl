# Model A/B, 2026-10-18: Qwen3.6-27B Q6_K vs Qwen3.8-27B UD-Q6_K_M

Same release (phase1-speed @ 371cd3e, `/opt/sever-tsl/releases/20261018-v5-ab-b`), same llama-server flags,
same video, no API jobs (API stopped for the test). Candidate: `unsloth/Qwen3.6-27B-GGUF` revision
82d411acf4a06cfb8d9b073a5211bf410bfc29bf, `Qwen3.6-27B-Q6_K.gguf` (22.5 GB) + `mmproj-F16.gguf`, downloaded
with publisher SHA-256 checks to `/opt/sever-tsl/models/qwen36-27b-q6`. Script: `deploy/model_ab_test.sh`.

Video 5 (bracket matching):

| | Qwen3.8-27B | Qwen3.6-27B |
|---|---|---|
| Wall time | 1,060 s | 1,841 s (1.74×) |
| Model calls | 166 | 199 |
| Output tokens | 41,104 | 79,579 (1.94×) |
| Labelled fields | 22/22 | 21/22 (`page-003 document_number` wrong) |
| Target types | 10/10 | 8/10 |
| Abstention checks | 22/22 | 22/22 |

Qwen3.6 writes Russian prose before its tool calls (field agent 325 vs 203 output tokens per call, joint review
863 vs 639, one call hit the 8,192-token cap) and spent 31–37 calls on three two-sided sheets where Qwen3.8 needed
11–15, ending them by the no-progress stop instead of "other document side". Decode speed is identical (~61 tok/s).
Despite its lead on MWS Vision Bench (Russian business-document KIE 0.893 vs 0.807), it is slower and less
accurate on this footage. Decision: stay on Qwen3.8-27B. Video 4 result to be appended.
