# before vs after

| | before | after | after/before |
|---|---|---|---|
| Wall time, s | 1,131.89 | 1,059.92 | 0.94 |
| Leaves | 15 | 15 | — |
| Seconds per leaf | 75.46 | 70.66 | — |
| Model requests | 188 | 166 | 0.88 |
| Requests per leaf | 12.53 | 11.07 | — |
| Model time, s | 1,002.56 | 927.36 | 0.93 |
| Input tokens | 1,992,133 | 1,600,314 | 0.8 |
| Output tokens | 41,696 | 41,104 | 0.99 |
| Calls: field_agent | 115 | 101 | |
| Calls: joint_review | 39 | 31 | |
| Calls: orientation | 15 | 15 | |
| Calls: triage | 19 | 19 | |

| Accuracy (provisional labels; matching bracket; address boundary before: current, after: current) | before | after |
|---|---|---|
| Labelled fields correct | 21/22 (95% 78–99%) | 22/22 (95% 85–100%) |
| Target types correct | 8/10 (95% 49–94%) | 10/10 (95% 72–100%) |
| Non-target rejected | 5/5 (95% 57–100%) | 5/5 (95% 57–100%) |
| Fully correct labelled sheets | 3/4 (95% 30–95%) | 4/4 (95% 51–100%) |
| Labelled intervals detected | 15/15 (95% 80–100%) | 15/15 (95% 80–100%) |
| Abstention checks passed (not-visible rows) | 22/22 (95% 85–100%) | 22/22 (95% 85–100%) |
| Paired labelled fields: before-only / after-only correct | 0 / 1 of 22 | McNemar p = 1.000 |
| Paired abstention checks: before-only / after-only correct | 0 / 0 of 22 | McNemar p = — |

Fields that changed:
- sample-05-page-004 consignor_address: before=wrong, after=ok

Matched through a boundary bracket (check by hand):
- before: sample-05-page-005 primary at 21.32 s, core 19.11–21.17 s
- after: sample-05-page-005 primary at 21.32 s, core 19.11–21.17 s

| Leaf | before requests | after requests | before s | after s | before stop | after stop |
|---|---|---|---|---|---|---|
| page-0001 | 13 | 9 | 52.56 | 44.72 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0002 | 8 | 8 | 31.01 | 29.45 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0003 | 22 | 14 | 105.16 | 98.07 | requires_other_document_side | requires_other_document_side |
| page-0004 | 10 | 12 | 54.67 | 65.01 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0005 | 15 | 9 | 59.41 | 32.51 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0006 | 19 | 15 | 109.45 | 104.56 | requires_other_document_side | requires_other_document_side |
| page-0007 | 3 | 3 | 5.92 | 5.77 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0008 | 3 | 3 | 6.02 | 6.02 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0009 | 3 | 3 | 5.73 | 5.73 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0010 | 3 | 3 | 5.97 | 5.86 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0011 | 8 | 8 | 21.26 | 23.3 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0012 | 15 | 11 | 100.83 | 77.66 | requires_other_document_side | requires_other_document_side |
| page-0013 | 13 | 11 | 96.76 | 84.76 | requires_other_document_side | requires_other_document_side |
| page-0014 | 19 | 42 | 107.48 | 293.2 | requires_other_document_side | requires_other_document_side |
| page-0015 | 34 | 15 | 301.23 | 114.23 | requires_other_document_side | requires_other_document_side |

after server timings over 166 responses: cached 1,087,515, processed 512,799 input tokens in 220,771.21 ms; generated 41,104 tokens in 675,451.24 ms.

- Timings include any other work on the model server during each run; check that no API jobs ran alongside.
