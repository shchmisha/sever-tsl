# main vs phase1-speed

| | main | phase1-speed | phase1-speed/main |
|---|---|---|---|
| Wall time, s | 2,665.37 | 1,134.49 | 0.43 |
| Leaves | 15 | 15 | — |
| Seconds per leaf | 177.69 | 75.63 | — |
| Model requests | 424 | 188 | 0.44 |
| Requests per leaf | 28.27 | 12.53 | — |
| Model time, s | 2,410.8 | 1,004.17 | 0.42 |
| Input tokens | 4,831,940 | 1,992,133 | 0.41 |
| Output tokens | 100,268 | 41,696 | 0.42 |
| Calls: field_agent | 294 | 115 | |
| Calls: joint_review | 55 | 39 | |
| Calls: orientation | 75 | 15 | |
| Calls: triage | 0 | 19 | |

| Accuracy (provisional labels) | main | phase1-speed |
|---|---|---|
| Labelled fields correct | 16/22 | 15/22 |
| Target types correct | 7/10 | 7/10 |
| Non-target rejected | 5/5 | 5/5 |
| Fully correct labelled sheets | 1/4 | 1/4 |
| Labelled intervals detected | 14/15 | 14/15 |

Fields that changed:
- sample-05-page-004 consignor_address: main=ok, phase1-speed=wrong

| Leaf | main requests | phase1-speed requests | main s | phase1-speed s | main stop | phase1-speed stop |
|---|---|---|---|---|---|---|
| page-0001 | 13 | 13 | 76.47 | 54.32 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0002 | 10 | 8 | 52.84 | 31.02 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0003 | 22 | 22 | 121.23 | 105.17 | requires_other_document_side | requires_other_document_side |
| page-0004 | 12 | 10 | 69.52 | 54.63 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0005 | 9 | 15 | 52.4 | 59.36 | all_required_fields_confirmed | all_required_fields_confirmed |
| page-0006 | 18 | 19 | 118.51 | 109.1 | requires_other_document_side | requires_other_document_side |
| page-0007 | 7 | 3 | 25.6 | 5.93 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0008 | 8 | 3 | 27.36 | 6.03 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0009 | 7 | 3 | 22.82 | 5.74 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0010 | 7 | 3 | 40.03 | 5.98 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0011 | 7 | 8 | 27.2 | 21.29 | out_of_scope_form_confirmed | out_of_scope_form_confirmed |
| page-0012 | 44 | 15 | 256.12 | 100.9 | requires_other_document_side | requires_other_document_side |
| page-0013 | 106 | 13 | 776.1 | 96.79 | source_views_exhausted | requires_other_document_side |
| page-0014 | 92 | 19 | 443.79 | 107.57 | requires_other_document_side | requires_other_document_side |
| page-0015 | 62 | 34 | 486.58 | 301.33 | requires_other_document_side | requires_other_document_side |

phase1-speed server timings over 188 responses: cached 1,352,407, processed 639,726 input tokens in 273,389.39 ms; generated 41,696 tokens in 688,474.45 ms.

- Timings include any other work on the model server during each run; check that no API jobs ran alongside.
