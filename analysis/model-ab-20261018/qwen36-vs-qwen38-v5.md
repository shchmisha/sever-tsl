# qwen38-27b-q6 vs qwen36-27b-q6

| | qwen38-27b-q6 | qwen36-27b-q6 | qwen36-27b-q6/qwen38-27b-q6 |
|---|---|---|---|
| Wall time, s | 1,059.92 | 1,841.18 | 1.74 |
| Leaves | 15 | 15 | — |
| Seconds per leaf | 70.66 | 122.75 | — |
| Model requests | 166 | 199 | 1.2 |
| Requests per leaf | 11.07 | 13.27 | — |
| Model time, s | 927.36 | 1,695.94 | 1.83 |
| Input tokens | 1,600,314 | 2,247,194 | 1.4 |
| Output tokens | 41,104 | 79,579 | 1.94 |
| Calls: field_agent | 101 | 122 | |
| Calls: joint_review | 31 | 42 | |
| Calls: orientation | 15 | 15 | |
| Calls: triage | 19 | 20 | |

| Accuracy (provisional labels; matching bracket; address boundary qwen38-27b-q6: current, qwen36-27b-q6: current) | qwen38-27b-q6 | qwen36-27b-q6 |
|---|---|---|
| Labelled fields correct | 22/22 (95% 85–100%) | 21/22 (95% 78–99%) |
| Target types correct | 10/10 (95% 72–100%) | 8/10 (95% 49–94%) |
| Non-target rejected | 5/5 (95% 57–100%) | 5/5 (95% 57–100%) |
| Fully correct labelled sheets | 4/4 (95% 51–100%) | 4/4 (95% 51–100%) |
| Labelled intervals detected | 15/15 (95% 80–100%) | 15/15 (95% 80–100%) |
| Abstention checks passed (not-visible rows) | 22/22 (95% 85–100%) | 22/22 (95% 85–100%) |
| Paired labelled fields: qwen38-27b-q6-only / qwen36-27b-q6-only correct | 1 / 0 of 22 | McNemar p = 1.000 |
| Paired abstention checks: qwen38-27b-q6-only / qwen36-27b-q6-only correct | 0 / 0 of 22 | McNemar p = — |

Fields that changed:
- sample-05-page-003 document_number: qwen38-27b-q6=ok, qwen36-27b-q6=wrong

Matched through a boundary bracket (check by hand):
- qwen38-27b-q6: sample-05-page-005 primary at 21.32 s, core 19.11–21.17 s
- qwen36-27b-q6: sample-05-page-005 primary at 21.32 s, core 19.11–21.17 s

| Leaf | qwen38-27b-q6 requests | qwen36-27b-q6 requests | qwen38-27b-q6 s | qwen36-27b-q6 s | qwen38-27b-q6 stop | qwen36-27b-q6 stop |
|---|---|---|---|---|---|---|
| page-0001 | 9 | 9 | 44.72 | 46.31 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0002 | 8 | 9 | 29.45 | 29.24 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0003 | 14 | 31 | 98.07 | 418.95 | requires_other_document_side | source_views_unread_after_no_progress |
| page-0004 | 12 | 6 | 65.01 | 36.99 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0005 | 9 | 13 | 32.51 | 64.31 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0006 | 15 | 36 | 104.56 | 353.62 | requires_other_document_side | source_views_unread_after_no_progress |
| page-0007 | 3 | 3 | 5.77 | 6.11 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0008 | 3 | 3 | 6.02 | 6.41 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0009 | 3 | 3 | 5.73 | 5.99 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0010 | 3 | 3 | 5.86 | 5.84 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0011 | 8 | 5 | 23.3 | 14.9 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0012 | 11 | 13 | 77.66 | 85.1 | requires_other_document_side | requires_other_document_side |
| page-0013 | 11 | 37 | 84.76 | 453.36 | requires_other_document_side | source_views_unread_after_no_progress |
| page-0014 | 42 | 13 | 293.2 | 107.94 | requires_other_document_side | requires_other_document_side |
| page-0015 | 15 | 15 | 114.23 | 137.4 | requires_other_document_side | requires_other_document_side |

qwen36-27b-q6 server timings over 199 responses: cached 1,447,018, processed 800,176 input tokens in 346,419.66 ms; generated 79,579 tokens in 1,303,171.83 ms.

- Timings include any other work on the model server during each run; check that no API jobs ran alongside.
