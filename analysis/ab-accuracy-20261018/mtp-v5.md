# mtp-off vs mtp-on

| | mtp-off | mtp-on | mtp-on/mtp-off |
|---|---|---|---|
| Wall time, s | 1,060.48 | 759.72 | 0.72 |
| Leaves | 15 | 15 | — |
| Seconds per leaf | 70.7 | 50.65 | — |
| Model requests | 166 | 171 | 1.03 |
| Requests per leaf | 11.07 | 11.4 | — |
| Model time, s | 927.07 | 626.25 | 0.68 |
| Input tokens | 1,600,314 | 1,739,835 | 1.09 |
| Output tokens | 41,104 | 44,667 | 1.09 |
| Calls: field_agent | 101 | 101 | |
| Calls: joint_review | 31 | 36 | |
| Calls: orientation | 15 | 15 | |
| Calls: triage | 19 | 19 | |

| Accuracy (provisional labels; matching bracket; address boundary mtp-off: current, mtp-on: current) | mtp-off | mtp-on |
|---|---|---|
| Labelled fields correct | 22/22 (95% 85–100%) | 22/22 (95% 85–100%) |
| Target types correct | 10/10 (95% 72–100%) | 10/10 (95% 72–100%) |
| Non-target rejected | 5/5 (95% 57–100%) | 5/5 (95% 57–100%) |
| Fully correct labelled sheets | 4/4 (95% 51–100%) | 4/4 (95% 51–100%) |
| Labelled intervals detected | 15/15 (95% 80–100%) | 15/15 (95% 80–100%) |
| Abstention checks passed (not-visible rows) | 22/22 (95% 85–100%) | 22/22 (95% 85–100%) |
| Paired labelled fields: mtp-off-only / mtp-on-only correct | 0 / 0 of 22 | McNemar p = — |
| Paired abstention checks: mtp-off-only / mtp-on-only correct | 0 / 0 of 22 | McNemar p = — |

Matched through a boundary bracket (check by hand):
- mtp-off: sample-05-page-005 primary at 21.32 s, core 19.11–21.17 s
- mtp-on: sample-05-page-005 primary at 21.32 s, core 19.11–21.17 s

| Leaf | mtp-off requests | mtp-on requests | mtp-off s | mtp-on s | mtp-off stop | mtp-on stop |
|---|---|---|---|---|---|---|
| page-0001 | 9 | 9 | 42.84 | 28.23 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0002 | 8 | 8 | 29.51 | 19.16 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0003 | 14 | 16 | 98.25 | 73.58 | requires_other_document_side | requires_other_document_side |
| page-0004 | 12 | 11 | 65.14 | 39 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0005 | 9 | 9 | 32.59 | 21.5 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0006 | 15 | 15 | 104.87 | 75.47 | requires_other_document_side | requires_other_document_side |
| page-0007 | 3 | 3 | 5.78 | 5.38 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0008 | 3 | 3 | 6.04 | 5.42 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0009 | 3 | 3 | 5.75 | 5.18 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0010 | 3 | 3 | 5.87 | 5.33 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0011 | 8 | 8 | 23.38 | 18.32 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0012 | 11 | 11 | 77.77 | 45.66 | requires_other_document_side | requires_other_document_side |
| page-0013 | 11 | 11 | 84.91 | 55.05 | requires_other_document_side | requires_other_document_side |
| page-0014 | 42 | 26 | 293.65 | 119.66 | requires_other_document_side | requires_other_document_side |
| page-0015 | 15 | 35 | 114.34 | 173.61 | requires_other_document_side | requires_other_document_side |

mtp-on server timings over 171 responses: cached 1,178,549, processed 561,286 input tokens in 250,304.75 ms; generated 44,667 tokens in 338,986.35 ms.

- Timings include any other work on the model server during each run; check that no API jobs ran alongside.
