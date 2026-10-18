# 61c7274 vs 8519b29

| | 61c7274 | 8519b29 | 8519b29/61c7274 |
|---|---|---|---|
| Wall time, s | 759.72 | 604.74 | 0.8 |
| Leaves | 15 | 15 | — |
| Seconds per leaf | 50.65 | 40.32 | — |
| Model requests | 171 | 147 | 0.86 |
| Requests per leaf | 11.4 | 9.8 | — |
| Model time, s | 626.25 | 491.86 | 0.79 |
| Input tokens | 1,739,835 | 1,450,861 | 0.83 |
| Output tokens | 44,667 | 33,280 | 0.74 |
| Calls: field_agent | 101 | 79 | |
| Calls: joint_review | 36 | 34 | |
| Calls: orientation | 15 | 15 | |
| Calls: triage | 19 | 19 | |

| Accuracy (provisional labels; matching bracket; address boundary 61c7274: current, 8519b29: current) | 61c7274 | 8519b29 |
|---|---|---|
| Labelled fields correct | 22/22 (95% 85–100%) | 22/22 (95% 85–100%) |
| Target types correct | 10/10 (95% 72–100%) | 9/10 (95% 60–98%) |
| Non-target rejected | 5/5 (95% 57–100%) | 5/5 (95% 57–100%) |
| Fully correct labelled sheets | 4/4 (95% 51–100%) | 4/4 (95% 51–100%) |
| Labelled intervals detected | 15/15 (95% 80–100%) | 15/15 (95% 80–100%) |
| Abstention checks passed (not-visible rows) | 22/22 (95% 85–100%) | 22/22 (95% 85–100%) |
| Paired labelled fields: 61c7274-only / 8519b29-only correct | 0 / 0 of 22 | McNemar p = — |
| Paired abstention checks: 61c7274-only / 8519b29-only correct | 0 / 0 of 22 | McNemar p = — |

Matched through a boundary bracket (check by hand):
- 61c7274: sample-05-page-005 primary at 21.32 s, core 19.11–21.17 s
- 8519b29: sample-05-page-005 primary at 21.32 s, core 19.11–21.17 s

| Leaf | 61c7274 requests | 8519b29 requests | 61c7274 s | 8519b29 s | 61c7274 stop | 8519b29 stop |
|---|---|---|---|---|---|---|
| page-0001 | 9 | 9 | 28.23 | 24.73 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0002 | 8 | 8 | 19.16 | 18.84 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0003 | 16 | 12 | 73.58 | 42.21 | requires_other_document_side | requires_other_document_side |
| page-0004 | 11 | 12 | 39 | 37.51 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0005 | 9 | 13 | 21.5 | 35.53 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0006 | 15 | 15 | 75.47 | 67.47 | requires_other_document_side | requires_other_document_side |
| page-0007 | 3 | 3 | 5.38 | 5.36 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0008 | 3 | 3 | 5.42 | 5.39 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0009 | 3 | 3 | 5.18 | 5.19 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0010 | 3 | 3 | 5.33 | 5.32 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0011 | 8 | 8 | 18.32 | 16.51 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0012 | 11 | 13 | 45.66 | 55.14 | requires_other_document_side | requires_other_document_side |
| page-0013 | 11 | 14 | 55.05 | 59.21 | requires_other_document_side | requires_other_document_side |
| page-0014 | 26 | 13 | 119.66 | 54.52 | requires_other_document_side | requires_other_document_side |
| page-0015 | 35 | 18 | 173.61 | 103.57 | requires_other_document_side | requires_other_document_side |

8519b29 server timings over 147 responses: cached 982,122, processed 468,739 input tokens in 207,694.99 ms; generated 33,280 tokens in 253,684.19 ms.

- Timings include any other work on the model server during each run; check that no API jobs ran alongside.
