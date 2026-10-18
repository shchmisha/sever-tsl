# 61c7274 vs 8519b29

| | 61c7274 | 8519b29 | 8519b29/61c7274 |
|---|---|---|---|
| Wall time, s | 2,086.05 | 1,089.26 | 0.52 |
| Leaves | 56 | 56 | — |
| Seconds per leaf | 37.25 | 19.45 | — |
| Model requests | 399 | 322 | 0.81 |
| Requests per leaf | 7.12 | 5.75 | — |
| Model time, s | 1,678.63 | 742.58 | 0.44 |
| Input tokens | 3,382,521 | 2,474,282 | 0.73 |
| Output tokens | 68,910 | 41,310 | 0.6 |
| Calls: field_agent | 195 | 136 | |
| Calls: joint_review | 50 | 35 | |
| Calls: orientation | 56 | 56 | |
| Calls: triage | 98 | 95 | |

| Accuracy (provisional labels; matching bracket; address boundary 61c7274: current, 8519b29: current) | 61c7274 | 8519b29 |
|---|---|---|
| Labelled fields correct | 22/22 (95% 85–100%) | 21/22 (95% 78–99%) |
| Target types correct | 10/10 (95% 72–100%) | 10/10 (95% 72–100%) |
| Non-target rejected | 42/43 (95% 88–100%) | 42/43 (95% 88–100%) |
| Fully correct labelled sheets | 4/4 (95% 51–100%) | 3/4 (95% 30–95%) |
| Labelled intervals detected | 52/53 (95% 90–100%) | 52/53 (95% 90–100%) |
| Abstention checks passed (not-visible rows) | 22/22 (95% 85–100%) | 22/22 (95% 85–100%) |
| Paired labelled fields: 61c7274-only / 8519b29-only correct | 1 / 0 of 22 | McNemar p = 1.000 |
| Paired abstention checks: 61c7274-only / 8519b29-only correct | 0 / 0 of 22 | McNemar p = — |

Fields that changed:
- sample-04-page-004 consignee: 61c7274=ok, 8519b29=wrong

Matched through a boundary bracket (check by hand):
- 61c7274: sample-04-page-007 primary at 31.99 s, core 29.89–31.42 s
- 61c7274: sample-04-page-023 primary at 82.04 s, core 80.40–81.94 s
- 61c7274: sample-04-page-027 primary at 96.96 s, core 97.26–98.80 s
- 61c7274: sample-04-page-030 primary at 108.16 s, core 106.46–107.99 s
- 8519b29: sample-04-page-007 primary at 31.99 s, core 29.89–31.42 s
- 8519b29: sample-04-page-023 primary at 82.04 s, core 80.40–81.94 s
- 8519b29: sample-04-page-027 primary at 96.96 s, core 97.26–98.80 s
- 8519b29: sample-04-page-030 primary at 108.16 s, core 106.46–107.99 s

| Leaf | 61c7274 requests | 8519b29 requests | 61c7274 s | 8519b29 s | 61c7274 stop | 8519b29 stop |
|---|---|---|---|---|---|---|
| page-0001 | 9 | 10 | 45.48 | 27.44 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0002 | 8 | 10 | 29.88 | 22.05 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0003 | 14 | 13 | 83.87 | 46.33 | requires_other_document_side | requires_other_document_side |
| page-0004 | 55 | 15 | 285.62 | 44.16 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0005 | 10 | 16 | 34.72 | 40.21 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0006 | 21 | 21 | 120.17 | 79.24 | requires_other_document_side | requires_other_document_side |
| page-0007 | 3 | 3 | 5.53 | 5.13 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0008 | 11 | 11 | 81.37 | 51.32 | requires_other_document_side | requires_other_document_side |
| page-0009 | 37 | 11 | 263.16 | 44.18 | source_views_unread_after_no_progress | requires_other_document_side |
| page-0010 | 37 | 12 | 266.85 | 49.67 | requires_other_document_side | requires_other_document_side |
| page-0011 | 33 | 26 | 254.51 | 98.31 | source_views_unread_after_no_progress | requires_other_document_side |
| page-0012 | 3 | 3 | 6.08 | 5.55 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0013 | 3 | 3 | 6.07 | 5.5 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0014 | 3 | 3 | 6.16 | 5.53 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0015 | 3 | 3 | 6.12 | 5.51 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0016 | 3 | 3 | 6.09 | 5.59 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0017 | 3 | 3 | 6.19 | 5.64 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0018 | 3 | 3 | 6.29 | 5.7 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0019 | 3 | 3 | 6.1 | 5.62 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0020 | 3 | 3 | 5.82 | 5.46 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0021 | 3 | 3 | 6.24 | 5.63 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0022 | 3 | 3 | 5.96 | 5.49 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0023 | 3 | 3 | 6.18 | 5.69 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0024 | 3 | 3 | 6.2 | 5.7 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0025 | 3 | 3 | 6.67 | 5.97 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0026 | 3 | 3 | 6.32 | 5.72 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0027 | 6 | 6 | 17.26 | 14.14 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0028 | 8 | 8 | 28.88 | 21.66 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0029 | 6 | 8 | 16.51 | 18.18 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0030 | 3 | 3 | 6.42 | 5.95 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0031 | 3 | 3 | 6.81 | 6.11 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0032 | 3 | 3 | 6.51 | 5.88 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0033 | 3 | 3 | 6.49 | 5.88 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0034 | 3 | 3 | 6.96 | 6.17 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0035 | 3 | 3 | 6.9 | 6.18 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0036 | 3 | 3 | 6.61 | 6.01 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0037 | 3 | 3 | 6.49 | 5.99 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0038 | 3 | 3 | 6.5 | 6.01 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0039 | 8 | 8 | 24.22 | 18.33 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0040 | 3 | 3 | 6.72 | 6.21 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0041 | 3 | 3 | 7.06 | 6.34 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0042 | 3 | 8 | 6.76 | 16.19 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0043 | 3 | 3 | 6.63 | 6.25 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0044 | 3 | 3 | 9.17 | 8.68 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0045 | 3 | 3 | 6.38 | 6.23 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0046 | 3 | 9 | 6.8 | 17.63 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0047 | 3 | 3 | 6.78 | 6.44 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0048 | 8 | 8 | 29.42 | 21.43 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0049 | 3 | 3 | 6.89 | 6.39 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0050 | 3 | 3 | 6.8 | 6.3 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0051 | 3 | 3 | 6.48 | 6.09 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0052 | 3 | 3 | 6.9 | 6.31 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0053 | 8 | 8 | 27.29 | 21.39 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0054 | 3 | 3 | 7.01 | 6.46 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0055 | 3 | 3 | 7.3 | 6.6 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0056 | 3 | 3 | 6.92 | 6.32 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |

8519b29 server timings over 322 responses: cached 1,631,430, processed 842,852 input tokens in 374,386.12 ms; generated 41,310 tokens in 317,282.23 ms.

- Timings include any other work on the model server during each run; check that no API jobs ran alongside.
