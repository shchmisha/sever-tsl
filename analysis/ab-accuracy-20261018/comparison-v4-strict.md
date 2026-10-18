# phase1-measured vs phase1-speed

| | phase1-measured | phase1-speed | phase1-speed/phase1-measured |
|---|---|---|---|
| Wall time, s | 1,900.77 | 2,119.31 | 1.11 |
| Leaves | 56 | 56 | — |
| Seconds per leaf | 33.94 | 37.84 | — |
| Model requests | 394 | 402 | 1.02 |
| Requests per leaf | 7.04 | 7.18 | — |
| Model time, s | 1,528.88 | 1,711.4 | 1.12 |
| Input tokens | 3,435,894 | 3,415,818 | 0.99 |
| Output tokens | 58,568 | 70,542 | 1.2 |
| Calls: field_agent | 187 | 195 | |
| Calls: joint_review | 54 | 53 | |
| Calls: orientation | 56 | 56 | |
| Calls: triage | 97 | 98 | |

| Accuracy (provisional labels; matching strict; address boundary phase1-measured: current, phase1-speed: current) | phase1-measured | phase1-speed |
|---|---|---|
| Labelled fields correct | 22/22 (95% 85–100%) | 20/22 (95% 72–97%) |
| Target types correct | 8/10 (95% 49–94%) | 10/10 (95% 72–100%) |
| Non-target rejected | 38/43 (95% 76–95%) | 38/43 (95% 76–95%) |
| Fully correct labelled sheets | 4/4 (95% 51–100%) | 3/4 (95% 30–95%) |
| Labelled intervals detected | 48/53 (95% 80–96%) | 48/53 (95% 80–96%) |
| Abstention checks passed (not-visible rows) | 22/22 (95% 85–100%) | 22/22 (95% 85–100%) |
| Paired labelled fields: phase1-measured-only / phase1-speed-only correct | 2 / 0 of 22 | McNemar p = 0.500 |
| Paired abstention checks: phase1-measured-only / phase1-speed-only correct | 0 / 0 of 22 | McNemar p = — |

Fields that changed:
- sample-04-page-004 consignee: phase1-measured=ok, phase1-speed=wrong
- sample-04-page-004 consignee_address: phase1-measured=ok, phase1-speed=wrong

| Leaf | phase1-measured requests | phase1-speed requests | phase1-measured s | phase1-speed s | phase1-measured stop | phase1-speed stop |
|---|---|---|---|---|---|---|
| page-0001 | 12 | 9 | 57.4 | 47.38 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0002 | 10 | 8 | 33.33 | 29.84 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0003 | 45 | 14 | 291.78 | 83.79 | source_views_unread_after_no_progress | requires_other_document_side |
| page-0004 | 14 | 58 | 61.48 | 318.63 | all_required_fields_confirmed | source_views_unread_after_no_progress |
| page-0005 | 14 | 10 | 43.68 | 34.63 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0006 | 19 | 21 | 101.82 | 119.99 | requires_other_document_side | requires_other_document_side |
| page-0007 | 3 | 3 | 5.49 | 5.52 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0008 | 26 | 11 | 115.31 | 81.3 | requires_other_document_side | requires_other_document_side |
| page-0009 | 43 | 37 | 298.63 | 262.86 | source_views_unread_after_no_progress | source_views_unread_after_no_progress |
| page-0010 | 20 | 37 | 114.01 | 266.64 | requires_other_document_side | requires_other_document_side |
| page-0011 | 20 | 33 | 147.07 | 254.25 | requires_other_document_side | source_views_unread_after_no_progress |
| page-0012 | 3 | 3 | 5.94 | 6.09 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0013 | 3 | 3 | 6.34 | 6.07 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0014 | 3 | 3 | 6.06 | 6.16 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0015 | 3 | 3 | 6.04 | 6.14 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0016 | 3 | 3 | 5.97 | 6.1 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0017 | 3 | 3 | 6.46 | 6.18 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0018 | 3 | 3 | 6.21 | 6.31 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0019 | 3 | 3 | 6.02 | 6.12 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0020 | 3 | 3 | 5.71 | 5.83 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0021 | 3 | 3 | 6.15 | 6.26 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0022 | 3 | 3 | 5.84 | 5.97 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0023 | 3 | 3 | 6.12 | 6.19 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0024 | 3 | 3 | 6.22 | 6.21 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0025 | 3 | 3 | 6.56 | 6.68 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0026 | 3 | 3 | 6.23 | 6.34 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0027 | 8 | 6 | 25.93 | 17.26 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0028 | 8 | 8 | 27.58 | 28.91 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0029 | 8 | 6 | 26.23 | 16.51 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0030 | 3 | 3 | 6.34 | 6.43 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0031 | 3 | 3 | 6.73 | 6.82 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0032 | 3 | 3 | 6.41 | 6.52 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0033 | 3 | 3 | 6.38 | 6.5 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0034 | 3 | 3 | 7.1 | 6.97 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0035 | 3 | 3 | 6.79 | 6.91 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0036 | 3 | 3 | 6.53 | 6.63 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0037 | 3 | 3 | 6.41 | 6.51 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0038 | 3 | 3 | 6.39 | 6.51 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0039 | 3 | 8 | 7.29 | 24.21 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0040 | 3 | 3 | 6.48 | 6.72 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0041 | 3 | 3 | 6.94 | 7.06 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0042 | 3 | 3 | 6.63 | 6.77 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0043 | 3 | 3 | 6.54 | 6.63 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0044 | 3 | 3 | 9.09 | 9.16 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0045 | 3 | 3 | 6.27 | 6.39 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0046 | 6 | 3 | 16.7 | 6.78 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0047 | 3 | 3 | 6.91 | 6.8 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0048 | 8 | 8 | 25.25 | 29.38 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0049 | 3 | 3 | 6.78 | 6.9 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0050 | 3 | 3 | 6.67 | 6.81 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0051 | 3 | 3 | 7.24 | 6.49 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0052 | 3 | 3 | 6.82 | 6.93 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0053 | 13 | 8 | 42.89 | 27.26 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0054 | 3 | 3 | 7.01 | 7 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0055 | 3 | 3 | 7.22 | 7.31 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0056 | 3 | 3 | 6.83 | 6.92 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |

phase1-speed server timings over 402 responses: cached 2,289,818, processed 1,126,000 input tokens in 481,484.75 ms; generated 70,542 tokens in 1,160,586.2 ms.

- Timings include any other work on the model server during each run; check that no API jobs ran alongside.
