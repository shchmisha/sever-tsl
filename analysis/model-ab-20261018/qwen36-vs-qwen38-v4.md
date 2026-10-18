# qwen38-27b-q6 vs qwen36-27b-q6

| | qwen38-27b-q6 | qwen36-27b-q6 | qwen36-27b-q6/qwen38-27b-q6 |
|---|---|---|---|
| Wall time, s | 2,119.31 | 2,394.96 | 1.13 |
| Leaves | 56 | 56 | — |
| Seconds per leaf | 37.84 | 42.77 | — |
| Model requests | 402 | 376 | 0.94 |
| Requests per leaf | 7.18 | 6.71 | — |
| Model time, s | 1,711.4 | 1,981.21 | 1.16 |
| Input tokens | 3,415,818 | 3,230,183 | 0.95 |
| Output tokens | 70,542 | 82,929 | 1.18 |
| Calls: field_agent | 195 | 174 | |
| Calls: joint_review | 53 | 51 | |
| Calls: orientation | 56 | 56 | |
| Calls: triage | 98 | 95 | |

| Accuracy (provisional labels; matching bracket; address boundary qwen38-27b-q6: current, qwen36-27b-q6: current) | qwen38-27b-q6 | qwen36-27b-q6 |
|---|---|---|
| Labelled fields correct | 20/22 (95% 72–97%) | 19/22 (95% 67–95%) |
| Target types correct | 10/10 (95% 72–100%) | 10/10 (95% 72–100%) |
| Non-target rejected | 42/43 (95% 88–100%) | 42/43 (95% 88–100%) |
| Fully correct labelled sheets | 3/4 (95% 30–95%) | 1/4 (95% 5–70%) |
| Labelled intervals detected | 52/53 (95% 90–100%) | 52/53 (95% 90–100%) |
| Abstention checks passed (not-visible rows) | 22/22 (95% 85–100%) | 22/22 (95% 85–100%) |
| Paired labelled fields: qwen38-27b-q6-only / qwen36-27b-q6-only correct | 2 / 1 of 22 | McNemar p = 1.000 |
| Paired abstention checks: qwen38-27b-q6-only / qwen36-27b-q6-only correct | 0 / 0 of 22 | McNemar p = — |

Fields that changed:
- sample-04-page-001 consignee_address: qwen38-27b-q6=ok, qwen36-27b-q6=wrong
- sample-04-page-004 consignee: qwen38-27b-q6=wrong, qwen36-27b-q6=ok
- sample-04-page-005 consignor: qwen38-27b-q6=ok, qwen36-27b-q6=wrong

Matched through a boundary bracket (check by hand):
- qwen38-27b-q6: sample-04-page-007 primary at 31.99 s, core 29.89–31.42 s
- qwen38-27b-q6: sample-04-page-023 primary at 82.04 s, core 80.40–81.94 s
- qwen38-27b-q6: sample-04-page-027 primary at 96.96 s, core 97.26–98.80 s
- qwen38-27b-q6: sample-04-page-030 primary at 108.16 s, core 106.46–107.99 s
- qwen36-27b-q6: sample-04-page-007 primary at 31.99 s, core 29.89–31.42 s
- qwen36-27b-q6: sample-04-page-023 primary at 82.04 s, core 80.40–81.94 s
- qwen36-27b-q6: sample-04-page-027 primary at 96.96 s, core 97.26–98.80 s
- qwen36-27b-q6: sample-04-page-030 primary at 108.16 s, core 106.46–107.99 s

| Leaf | qwen38-27b-q6 requests | qwen36-27b-q6 requests | qwen38-27b-q6 s | qwen36-27b-q6 s | qwen38-27b-q6 stop | qwen36-27b-q6 stop |
|---|---|---|---|---|---|---|
| page-0001 | 9 | 18 | 47.38 | 128.52 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0002 | 8 | 12 | 29.84 | 41.07 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0003 | 14 | 20 | 83.79 | 141.92 | requires_other_document_side | requires_other_document_side |
| page-0004 | 58 | 33 | 318.63 | 235.26 | source_views_unread_after_no_progress | all_required_fields_confirmed_after_joint_review |
| page-0005 | 10 | 9 | 34.63 | 66.6 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0006 | 21 | 11 | 119.99 | 146.16 | requires_other_document_side | requires_other_document_side |
| page-0007 | 3 | 3 | 5.52 | 5.83 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0008 | 11 | 17 | 81.3 | 107.91 | requires_other_document_side | requires_other_document_side |
| page-0009 | 37 | 42 | 262.86 | 430.19 | source_views_unread_after_no_progress | source_views_unread_after_no_progress |
| page-0010 | 37 | 14 | 266.64 | 95.44 | requires_other_document_side | requires_other_document_side |
| page-0011 | 33 | 38 | 254.25 | 338.08 | source_views_unread_after_no_progress | source_views_unread_after_no_progress |
| page-0012 | 3 | 3 | 6.09 | 6.48 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0013 | 3 | 3 | 6.07 | 6.81 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0014 | 3 | 3 | 6.16 | 6.53 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0015 | 3 | 3 | 6.14 | 6.49 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0016 | 3 | 3 | 6.1 | 6.36 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0017 | 3 | 3 | 6.18 | 6.95 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0018 | 3 | 7 | 6.31 | 30.28 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0019 | 3 | 3 | 6.12 | 6.51 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0020 | 3 | 3 | 5.83 | 6.96 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0021 | 3 | 3 | 6.26 | 6.63 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0022 | 3 | 3 | 5.97 | 6.35 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0023 | 3 | 3 | 6.19 | 6.61 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0024 | 3 | 3 | 6.21 | 6.5 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0025 | 3 | 3 | 6.68 | 7.04 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0026 | 3 | 3 | 6.34 | 6.69 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0027 | 6 | 6 | 17.26 | 28.5 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0028 | 8 | 6 | 28.91 | 23.37 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0029 | 6 | 6 | 16.51 | 20.33 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0030 | 3 | 3 | 6.43 | 6.8 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0031 | 3 | 3 | 6.82 | 6.44 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0032 | 3 | 3 | 6.52 | 6.9 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0033 | 3 | 3 | 6.5 | 6.92 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0034 | 3 | 3 | 6.97 | 6.89 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0035 | 3 | 3 | 6.91 | 6.9 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0036 | 3 | 3 | 6.63 | 7.01 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0037 | 3 | 3 | 6.51 | 6.79 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0038 | 3 | 3 | 6.51 | 6.66 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0039 | 8 | 6 | 24.21 | 23.06 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0040 | 3 | 3 | 6.72 | 7 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0041 | 3 | 3 | 7.06 | 7.47 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0042 | 3 | 3 | 6.77 | 6.97 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0043 | 3 | 3 | 6.63 | 7.1 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0044 | 3 | 3 | 9.16 | 9.6 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0045 | 3 | 3 | 6.39 | 6.76 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0046 | 3 | 5 | 6.78 | 14.71 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0047 | 3 | 3 | 6.8 | 7.09 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0048 | 8 | 6 | 29.38 | 17.76 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0049 | 3 | 3 | 6.9 | 7.01 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0050 | 3 | 3 | 6.81 | 7.17 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0051 | 3 | 3 | 6.49 | 7.23 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0052 | 3 | 3 | 6.93 | 7.27 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0053 | 8 | 6 | 27.26 | 21.4 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0054 | 3 | 3 | 7 | 7.34 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0055 | 3 | 3 | 7.31 | 7.65 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0056 | 3 | 3 | 6.92 | 7.25 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |

qwen36-27b-q6 server timings over 376 responses: cached 1,972,131, processed 1,258,052 input tokens in 554,271.45 ms; generated 82,929 tokens in 1,356,175.95 ms.

- Timings include any other work on the model server during each run; check that no API jobs ran alongside.
