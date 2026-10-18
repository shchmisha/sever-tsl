# Per-document export and accuracy, 2026-10-18

The customer reviewed the output against the technical specification (section 3.1.4: a result *per processed
document*) and the marked-up example PDF «Считываемые поля документов». The 19 fields matched, but the export
listed every sheet showing: copies, reverse sides, continuation pages and repeat showings each became a set of
rows, most of them empty ("unreadable" on sheets that do not carry the field). One video with two documents
produced 12 "documents" and 58 field rows, 30 of them empty.

Commit 7f9e029 joins sheets into documents (form + own number; see `sever/documents.py` POLICY) and exports
`documents.csv` (one row per document, one column per contract field), `documents.json` and a report that leads
with the document table. Accuracy is now scored per document (`evaluate_documents`, `tools/score_documents.py`);
the ground truth has explicit `document_id` on every labelled target sheet (labels unchanged).

## Results so far (production release 61c7274 readings; MTP on for sample-05 and set2 videos)

See `score-4-videos.md`. Pooled over 4 videos, 8 labelled documents:

| | Result |
|---|---|
| Documents found | 8/8 |
| Document type correct | 8/8 |
| Labelled fields correct | 38/39 |
| Values shown in the table that are correct | 37/37 |
| Documents fully correct | 7/8 |
| Extra documents in the output | 1 |

Errors found only at document level, in set2-20260821_145110:
- the own number РРРБ-128027 is read as РРБ-128027 (a repeated letter dropped) on 5 of 12 ТОРГ-12 sheets and
  2 of 5 ТрН sheets, and once each as РРРВ / PРРВ (Latin P) / ПРРБ; assembly joins these readings (same digits,
  look-alike letters) and keeps the majority, so ТОРГ-12 is right and ТрН is reported as undetermined (2 vs 2);
- a plain счёт-фактура (no УПД title or transfer section visible) is typed УПД and becomes an extra document;
- the video is cut into 36 sheets where the labels have 15, which also made the run slow (3,206 s).

## Final measurement (all 7 labelled videos, release 8519b29)

See `final-7-videos/`: per document 17/18 documents found, 70/75 labelled fields, 12/14 fully labelled documents
fully correct, 4 extra documents; on the 4 videos above again 38/39 fields and 7/8 documents. The Russian final
report with error statistics by cause is `analysis/final-report-20261018/`.
