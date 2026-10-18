# before-fix vs after-fix

| | before-fix | after-fix | after-fix/before-fix |
|---|---|---|---|
| Wall time, s | 2,122.29 | 2,086.05 | 0.98 |
| Leaves | 56 | 56 | — |
| Seconds per leaf | 37.9 | 37.25 | — |
| Model requests | 402 | 399 | 0.99 |
| Requests per leaf | 7.18 | 7.12 | — |
| Model time, s | 1,712.67 | 1,678.63 | 0.98 |
| Input tokens | 3,415,818 | 3,382,521 | 0.99 |
| Output tokens | 70,542 | 68,910 | 0.98 |
| Calls: field_agent | 195 | 195 | |
| Calls: joint_review | 53 | 50 | |
| Calls: orientation | 56 | 56 | |
| Calls: triage | 98 | 98 | |

| Accuracy (provisional labels; matching bracket; address boundary before-fix: current, after-fix: current) | before-fix | after-fix |
|---|---|---|
| Labelled fields correct | 20/22 (95% 72–97%) | 22/22 (95% 85–100%) |
| Target types correct | 10/10 (95% 72–100%) | 10/10 (95% 72–100%) |
| Non-target rejected | 42/43 (95% 88–100%) | 42/43 (95% 88–100%) |
| Fully correct labelled sheets | 3/4 (95% 30–95%) | 4/4 (95% 51–100%) |
| Labelled intervals detected | 52/53 (95% 90–100%) | 52/53 (95% 90–100%) |
| Abstention checks passed (not-visible rows) | 22/22 (95% 85–100%) | 22/22 (95% 85–100%) |
| Paired labelled fields: before-fix-only / after-fix-only correct | 0 / 2 of 22 | McNemar p = 0.500 |
| Paired abstention checks: before-fix-only / after-fix-only correct | 0 / 0 of 22 | McNemar p = — |

Fields that changed:
- sample-04-page-004 consignee: before-fix=wrong, after-fix=ok
- sample-04-page-004 consignee_address: before-fix=wrong, after-fix=ok

Matched through a boundary bracket (check by hand):
- before-fix: sample-04-page-007 primary at 31.99 s, core 29.89–31.42 s
- before-fix: sample-04-page-023 primary at 82.04 s, core 80.40–81.94 s
- before-fix: sample-04-page-027 primary at 96.96 s, core 97.26–98.80 s
- before-fix: sample-04-page-030 primary at 108.16 s, core 106.46–107.99 s
- after-fix: sample-04-page-007 primary at 31.99 s, core 29.89–31.42 s
- after-fix: sample-04-page-023 primary at 82.04 s, core 80.40–81.94 s
- after-fix: sample-04-page-027 primary at 96.96 s, core 97.26–98.80 s
- after-fix: sample-04-page-030 primary at 108.16 s, core 106.46–107.99 s

| Leaf | before-fix requests | after-fix requests | before-fix s | after-fix s | before-fix stop | after-fix stop |
|---|---|---|---|---|---|---|
| page-0001 | 9 | 9 | 45.65 | 45.48 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0002 | 8 | 8 | 29.88 | 29.88 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0003 | 14 | 14 | 83.89 | 83.87 | requires_other_document_side | requires_other_document_side |
| page-0004 | 58 | 55 | 319.42 | 285.62 | source_views_unread_after_no_progress | all_required_fields_confirmed |
| page-0005 | 10 | 10 | 34.71 | 34.72 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0006 | 21 | 21 | 120.21 | 120.17 | requires_other_document_side | requires_other_document_side |
| page-0007 | 3 | 3 | 5.55 | 5.53 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0008 | 11 | 11 | 81.43 | 81.37 | requires_other_document_side | requires_other_document_side |
| page-0009 | 37 | 37 | 263.4 | 263.16 | source_views_unread_after_no_progress | source_views_unread_after_no_progress |
| page-0010 | 37 | 37 | 267.11 | 266.85 | requires_other_document_side | requires_other_document_side |
| page-0011 | 33 | 33 | 254.77 | 254.51 | source_views_unread_after_no_progress | source_views_unread_after_no_progress |
| page-0012 | 3 | 3 | 6.13 | 6.08 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0013 | 3 | 3 | 6.08 | 6.07 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0014 | 3 | 3 | 6.19 | 6.16 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0015 | 3 | 3 | 6.14 | 6.12 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0016 | 3 | 3 | 6.12 | 6.09 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0017 | 3 | 3 | 6.21 | 6.19 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0018 | 3 | 3 | 6.32 | 6.29 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0019 | 3 | 3 | 6.15 | 6.1 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0020 | 3 | 3 | 5.86 | 5.82 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0021 | 3 | 3 | 6.27 | 6.24 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0022 | 3 | 3 | 6.01 | 5.96 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0023 | 3 | 3 | 6.21 | 6.18 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0024 | 3 | 3 | 6.22 | 6.2 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0025 | 3 | 3 | 6.7 | 6.67 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0026 | 3 | 3 | 6.35 | 6.32 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0027 | 6 | 6 | 17.3 | 17.26 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0028 | 8 | 8 | 29.02 | 28.88 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0029 | 6 | 6 | 16.56 | 16.51 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0030 | 3 | 3 | 6.46 | 6.42 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0031 | 3 | 3 | 6.84 | 6.81 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0032 | 3 | 3 | 6.53 | 6.51 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0033 | 3 | 3 | 6.51 | 6.49 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0034 | 3 | 3 | 6.99 | 6.96 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0035 | 3 | 3 | 6.93 | 6.9 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0036 | 3 | 3 | 6.64 | 6.61 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0037 | 3 | 3 | 6.52 | 6.49 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0038 | 3 | 3 | 6.53 | 6.5 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0039 | 8 | 8 | 24.26 | 24.22 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0040 | 3 | 3 | 6.73 | 6.72 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0041 | 3 | 3 | 7.1 | 7.06 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0042 | 3 | 3 | 6.77 | 6.76 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0043 | 3 | 3 | 6.66 | 6.63 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0044 | 3 | 3 | 9.24 | 9.17 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0045 | 3 | 3 | 6.41 | 6.38 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0046 | 3 | 3 | 6.83 | 6.8 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0047 | 3 | 3 | 6.82 | 6.78 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0048 | 8 | 8 | 29.43 | 29.42 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0049 | 3 | 3 | 6.91 | 6.89 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0050 | 3 | 3 | 6.81 | 6.8 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0051 | 3 | 3 | 6.49 | 6.48 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0052 | 3 | 3 | 6.94 | 6.9 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0053 | 8 | 8 | 27.32 | 27.29 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0054 | 3 | 3 | 7.01 | 7.01 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0055 | 3 | 3 | 7.32 | 7.3 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0056 | 3 | 3 | 6.94 | 6.92 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |

after-fix server timings over 399 responses: cached 2,277,167, processed 1,105,354 input tokens in 475,460.43 ms; generated 68,910 tokens in 1,133,729.45 ms.

- Timings include any other work on the model server during each run; check that no API jobs ran alongside.
