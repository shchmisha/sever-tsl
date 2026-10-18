"""One record per document: sheets of the same form and own number are joined.

A video shows physical sheets, often more than once: copies, reverse sides,
continuation pages and repeat showings. The technical specification asks for a
result per processed document, so sheet readings are joined into documents
here. Joining never changes a reading: every value comes unchanged from a sheet
reading, the sheets behind each value are listed, and disagreements between
sheets are reported instead of being resolved silently.
"""
from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

from .normalize import comparison, text
from .schema import FIELDS_BY_TYPE

DOCUMENT_ASSEMBLY_VERSION = "documents-by-type-and-own-number-v3-lookalike-letters"

NUMBER_FIELD = {"upd": "invoice_number", "torg12": "document_number", "ttn": "document_number", "trn": "document_number"}
DATE_FIELD = {"upd": "invoice_date", "torg12": "document_date", "ttn": "document_date", "trn": "document_date"}
TYPE_LABELS = {"upd": "УПД", "torg12": "ТОРГ-12", "ttn": "ТТН", "trn": "ТрН"}
FIELD_LABELS = {
    "upd": {"invoice_number": "Номер счёта-фактуры", "invoice_date": "Дата счёта-фактуры",
            "consignor": "Грузоотправитель", "consignor_address": "Адрес грузоотправителя",
            "consignee": "Грузополучатель", "consignee_address": "Адрес грузополучателя"},
    "torg12": {"supplier": "Поставщик", "consignee": "Грузополучатель",
               "document_number": "Номер документа", "document_date": "Дата составления"},
    "ttn": {"consignor": "Грузоотправитель", "consignee": "Грузополучатель",
            "document_number": "Номер документа", "document_date": "Дата составления"},
    "trn": {"document_date": "Дата транспортной накладной", "document_number": "Номер транспортной накладной",
            "consignor": "Грузоотправитель", "customer": "Заказчик услуг по перевозке",
            "consignee": "Грузополучатель"},
}
# One table for all four forms: each contract field has exactly one column.
COLUMNS = (("number", "Номер"), ("date", "Дата"), ("consignor", "Грузоотправитель"),
           ("consignor_address", "Адрес грузоотправителя"), ("supplier", "Поставщик"),
           ("customer", "Заказчик услуг по перевозке"), ("consignee", "Грузополучатель"),
           ("consignee_address", "Адрес грузополучателя"))
FIELD_COLUMN = {"invoice_number": "number", "document_number": "number", "invoice_date": "date",
                "document_date": "date", "consignor": "consignor", "consignor_address": "consignor_address",
                "supplier": "supplier", "customer": "customer", "consignee": "consignee",
                "consignee_address": "consignee_address"}
BLANK_TEXT = "(не заполнено)"
ASSIGNMENT_REASONS = {
    "own_number": "собственный номер документа на листе",
    "own_number_variant": "номер на листе отличается от номера документа только похожими буквами",
    "continues_previous_sheet": "лист без номера сразу после листа этого документа",
    "precedes_next_sheet": "лист без номера сразу перед листом этого документа",
    "only_document_of_this_type": "единственный документ этого типа в видео",
    "number_not_read": "номер не прочитан; лист с данными выделен в отдельный документ",
}
POLICY = {
    "version": DOCUMENT_ASSEMBLY_VERSION,
    "identity": "document type plus its own number as read on the sheet, compared after formatting-only normalization",
    "number_variants": "readings with the same digits whose letters differ by at most two look-alike or repeated-letter "
                       "edits are one document unless their dates disagree; the number shown is the reading on most sheets",
    "sheets_without_number": "joined to the document of the same type shown immediately before, else immediately after, "
                             "else to the only document of that type in the video; a sheet with values and no such "
                             "document becomes its own document, a sheet with no values is listed as unassigned",
    "field_value": "taken unchanged from the sheets that read it; when sheets disagree the value supported by more "
                   "sheets is kept and the others are listed; equal support is reported as a conflict without a value",
    "unread_on_a_sheet": "a sheet that does not show a field does not vote; only recognized and printed-blank readings count",
    "out_of_scope_forms": "sheets of forms outside the four contract types are not documents and are listed separately",
}


# Latin capitals that print like Cyrillic ones; used only to decide whether two
# number readings can be the same document, never to change a reading.
_LOOKALIKE = str.maketrans("ABCEHKMOPTXYabcehkmoptxy", "АВСЕНКМОРТХУавсенкмортху")


def _number_shape(key: str) -> tuple[str, str]:
    folded = key.translate(_LOOKALIKE).casefold()
    return "".join(ch for ch in folded if ch.isdigit()), "".join(ch for ch in folded if ch.isalpha())


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def number_variants(left: str, right: str) -> bool:
    """Two number readings of one document: same digits, letters differing by at most two look-alike edits."""
    left_digits, left_letters = _number_shape(left)
    right_digits, right_letters = _number_shape(right)
    return (len(left_digits) >= 3 and left_digits == right_digits
            and _edit_distance(left_letters, right_letters) <= 2)


def _key(field: str, value) -> str | None:
    """Comparison key for joining sheets: formatting-only differences and look-alike letters agree."""
    if not isinstance(value, str) or not value.strip():
        return None
    return (comparison(field, value) or text(value)).casefold().translate(_LOOKALIKE)


def _frames(field: dict) -> int:
    return len({evidence.get("frame_id") for evidence in field.get("evidence", []) if evidence.get("frame_id")})


def _sheet_summary(page: dict, assignment: str | None) -> dict:
    fields = page["result"].get("fields") or {}
    return {"page_id": page.get("page_id"), "start_s": page.get("start_s"), "end_s": page.get("end_s"),
            "document_type": page["result"].get("document_type"), "assignment": assignment,
            "fields_with_values": [name for name, field in fields.items()
                                   if field.get("status") in {"recognized", "not_present"}],
            "errors": list(page["result"].get("errors", []))}


def _merge_field(name: str, sheets: list[dict]) -> dict:
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    statuses = []
    for sheet in sheets:
        field = (sheet["page"]["result"].get("fields") or {}).get(name) or {}
        status = field.get("status")
        statuses.append(status)
        if status == "recognized" and field.get("raw_value"):
            groups.setdefault(("recognized", _key(name, field["raw_value"])), []).append((sheet, field))
        elif status == "not_present":
            groups.setdefault(("not_present", None), []).append((sheet, field))
    if not groups:
        status = "conflict" if "conflict" in statuses else "unreadable"
        return {"status": status, "raw_value": None, "normalized_value": None, "source_page_ids": [],
                "candidates": [], "reason": ("sheets of this document disagree and no sheet confirmed a value"
                                             if status == "conflict" else "no sheet of this document shows a readable value")}
    ranked = sorted(groups.values(), key=lambda members: -len(members))
    candidates = [{"status": members[0][1]["status"], "raw_value": members[0][1].get("raw_value"),
                   "source_page_ids": [sheet["page"]["page_id"] for sheet, _ in members]} for members in ranked]
    if len(ranked) > 1 and len(ranked[1]) == len(ranked[0]):
        return {"status": "conflict", "raw_value": None, "normalized_value": None, "source_page_ids": [],
                "candidates": candidates,
                "reason": "sheets of this document read different values with equal support"}
    best = ranked[0]
    # The sheet whose value rests on the most distinct frames supplies the literal text; the earliest wins ties.
    sheet, field = max(best, key=lambda item: (_frames(item[1]), -best.index(item)))
    return {"status": field["status"], "raw_value": field.get("raw_value"),
            "normalized_value": field.get("normalized_value"),
            "source_page_ids": [member["page"]["page_id"] for member, _ in best],
            "candidates": candidates if len(ranked) > 1 else [],
            "reason": ("sheets disagree; the value read on more sheets is kept and the others are listed"
                       if len(ranked) > 1 else "read on the listed sheets")}


def assemble_documents(result: dict) -> dict:
    """Join sheet results of one video into documents (pure; reads ``result['pages']`` only)."""
    pages = [page for page in result.get("pages", []) if isinstance(page.get("result"), dict)]
    pages.sort(key=lambda page: (page.get("start_s") if isinstance(page.get("start_s"), (int, float)) else 0.0,
                                 str(page.get("page_id"))))
    sheets, other_forms = [], []
    for page in pages:
        kind = page["result"].get("document_type")
        if kind not in NUMBER_FIELD:
            recognition = page.get("image_recognition") or {}
            views = recognition.get("views") or []
            other_forms.append({"page_id": page.get("page_id"), "start_s": page.get("start_s"), "end_s": page.get("end_s"),
                                "observed_form": next((view.get("observed_form") for view in views if view.get("observed_form")), None),
                                "reason": recognition.get("stop_reason")})
            continue
        number = (page["result"].get("fields") or {}).get(NUMBER_FIELD[kind]) or {}
        key = _key(NUMBER_FIELD[kind], number.get("raw_value")) if number.get("status") == "recognized" else None
        has_values = any(field.get("status") in {"recognized", "not_present"}
                         for field in (page["result"].get("fields") or {}).values())
        sheets.append({"page": page, "type": kind, "key": key, "has_values": has_values, "doc": None, "assignment": None})

    documents: "OrderedDict[tuple, dict]" = OrderedDict()

    def open_document(identity):
        documents[identity] = {"type": identity[0], "sheets": []}
        return identity

    for sheet in sheets:
        if sheet["key"] is not None:
            identity = (sheet["type"], "number", sheet["key"])
            if identity not in documents:
                open_document(identity)
            documents[identity]["sheets"].append(sheet)
            sheet["doc"], sheet["assignment"] = identity, "own_number"
    # Readings of one number that differ only in look-alike or repeated letters
    # (РРРБ-128027 / РРБ-128027 / PРРВ-128027) are one document, unless their dates disagree.
    def group_date(identity):
        keys = [(_key(DATE_FIELD[identity[0]], (sheet["page"]["result"].get("fields") or {}).get(DATE_FIELD[identity[0]], {}).get("raw_value"))
                 if (sheet["page"]["result"].get("fields") or {}).get(DATE_FIELD[identity[0]], {}).get("status") == "recognized" else None)
                for sheet in documents[identity]["sheets"]]
        keys = [key for key in keys if key]
        return max(set(keys), key=keys.count) if keys else None

    parent = {identity: identity for identity in documents}

    def root(identity):
        while parent[identity] != identity:
            identity = parent[identity]
        return identity

    identities = list(documents)
    for i, left in enumerate(identities):
        for right in identities[i + 1:]:
            if left[0] != right[0] or root(left) == root(right) or not number_variants(left[2], right[2]):
                continue
            left_date, right_date = group_date(left), group_date(right)
            if left_date and right_date and left_date != right_date:
                continue
            a, b = root(left), root(right)
            keep, drop = (a, b) if len(documents[a]["sheets"]) >= len(documents[b]["sheets"]) else (b, a)
            parent[drop] = keep
    if any(parent[identity] != identity for identity in identities):
        joined: "OrderedDict[tuple, dict]" = OrderedDict()
        for identity in identities:
            target = root(identity)
            record = joined.setdefault(target, {"type": identity[0], "sheets": []})
            for sheet in documents[identity]["sheets"]:
                sheet["doc"] = target
                if identity != target:
                    sheet["assignment"] = "own_number_variant"
                record["sheets"].append(sheet)
        documents.clear()
        documents.update(joined)
    numbered = {identity for identity in documents}
    unassigned = []
    for index, sheet in enumerate(sheets):
        if sheet["doc"] is not None:
            continue
        previous = sheets[index - 1] if index > 0 else None
        following = sheets[index + 1] if index + 1 < len(sheets) else None
        same_type = [identity for identity in numbered if identity[0] == sheet["type"]]
        if previous and previous["type"] == sheet["type"] and previous["doc"] is not None:
            target, reason = previous["doc"], "continues_previous_sheet"
        elif following and following["type"] == sheet["type"] and following["doc"] is not None:
            target, reason = following["doc"], "precedes_next_sheet"
        elif len(same_type) == 1:
            target, reason = same_type[0], "only_document_of_this_type"
        elif sheet["has_values"]:
            target, reason = open_document((sheet["type"], "sheet", sheet["page"].get("page_id"))), "number_not_read"
        else:
            unassigned.append(_sheet_summary(sheet["page"], None) | {
                "reason": "no own number, no field values and no neighbouring sheet of the same form"})
            continue
        documents[target]["sheets"].append(sheet)
        sheet["doc"], sheet["assignment"] = target, reason

    records = []
    for position, (identity, group) in enumerate(documents.items(), start=1):
        kind = group["type"]
        members = sorted(group["sheets"], key=lambda sheet: (sheet["page"].get("start_s") or 0.0, str(sheet["page"].get("page_id"))))
        fields = {name: _merge_field(name, members) for name in FIELDS_BY_TYPE[kind]}
        undetermined = [name for name, field in fields.items() if field["status"] in {"unreadable", "conflict"}]
        starts = [sheet["page"].get("start_s") for sheet in members if isinstance(sheet["page"].get("start_s"), (int, float))]
        ends = [sheet["page"].get("end_s") for sheet in members if isinstance(sheet["page"].get("end_s"), (int, float))]
        records.append({
            "document_id": f"doc-{position:03d}", "document_type": kind, "document_type_label": TYPE_LABELS[kind],
            "number": fields[NUMBER_FIELD[kind]]["raw_value"], "date": fields[DATE_FIELD[kind]]["raw_value"],
            "number_readings_joined": list(OrderedDict.fromkeys(
                field["raw_value"] for field in ((sheet["page"]["result"].get("fields") or {}).get(NUMBER_FIELD[kind]) or {}
                                                 for sheet in members)
                if field.get("status") == "recognized" and field.get("raw_value"))),
            "status": "complete" if not undetermined else "incomplete", "undetermined_fields": undetermined,
            "fields": fields,
            "sheets": [_sheet_summary(sheet["page"], sheet["assignment"]) for sheet in members],
            "start_s": min(starts) if starts else None, "end_s": max(ends) if ends else None,
            "errors": sorted({error for sheet in members for error in sheet["page"]["result"].get("errors", [])}),
        })
    by_type = {}
    for record in records:
        by_type[record["document_type"]] = by_type.get(record["document_type"], 0) + 1
    return {"version": DOCUMENT_ASSEMBLY_VERSION, "policy": POLICY, "documents": records,
            "unassigned_sheets": unassigned, "other_forms": other_forms,
            "summary": {"documents": len(records), "complete_documents": sum(r["status"] == "complete" for r in records),
                        "documents_by_type": by_type, "sheets_total": len(pages), "sheets_in_documents":
                        sum(len(r["sheets"]) for r in records), "unassigned_sheets": len(unassigned),
                        "other_form_sheets": len(other_forms)}}


def _cell(field: dict) -> str:
    if field["status"] == "recognized":
        return field["raw_value"] or ""
    if field["status"] == "not_present":
        return BLANK_TEXT
    return ""


def undetermined_text(record: dict) -> str:
    labels = FIELD_LABELS[record["document_type"]]
    parts = []
    for name in record["undetermined_fields"]:
        field = record["fields"][name]
        if field["status"] == "conflict" and field["candidates"]:
            values = " / ".join(candidate["raw_value"] or BLANK_TEXT for candidate in field["candidates"])
            parts.append(f"{labels[name]}: разные значения на листах ({values})")
        elif field["status"] == "conflict":
            parts.append(f"{labels[name]}: кадры расходятся")
        else:
            parts.append(f"{labels[name]}: не читается")
    return "; ".join(parts)


def document_row(record: dict) -> dict:
    """Customer table row: one document, one column per contract field."""
    row = {"№": record["document_id"].removeprefix("doc-").lstrip("0") or "0",
           "Тип документа": record["document_type_label"]}
    row.update({title: "" for _, title in COLUMNS})
    titles = dict(COLUMNS)
    for name, field in record["fields"].items():
        row[titles[FIELD_COLUMN[name]]] = _cell(field)
    row["Не удалось определить"] = undetermined_text(record)
    row["Листы в видео, с"] = "; ".join(f"{sheet['start_s']:.1f}–{sheet['end_s']:.1f}" for sheet in record["sheets"]
                                       if isinstance(sheet.get("start_s"), (int, float)) and isinstance(sheet.get("end_s"), (int, float)))
    return row


DOCUMENT_CSV_COLUMNS = ["№", "Тип документа", *[title for _, title in COLUMNS], "Не удалось определить", "Листы в видео, с"]


def export_documents(out: Path, assembly: dict) -> None:
    """Write documents.csv (customer table) and documents.json (full detail)."""
    import json

    out = Path(out)
    with (out / "documents.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=DOCUMENT_CSV_COLUMNS)
        writer.writeheader()
        for record in assembly["documents"]:
            writer.writerow(document_row(record))
    temporary = out / "documents.json.tmp"
    temporary.write_text(json.dumps(assembly, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(out / "documents.json")


STATUS_TEXT = {"recognized": "прочитано", "not_present": "поле на бланке не заполнено",
               "unreadable": "не удалось прочитать", "conflict": "листы расходятся"}


def document_title(record: dict) -> str:
    """«УПД № 2591 от 23 августа 2026 г.» with honest gaps when the number or date is unknown."""
    number = record.get("number")
    date = record.get("date")
    title = record["document_type_label"] + (f" № {number}" if number else " без определённого номера")
    return title + (f" от {date}" if date else "")


def _short_form(text: str | None) -> str:
    value = (text or "").split(";")[0].strip()
    return value if len(value) <= 70 else value[:67].rstrip() + "…"


def customer_summary(assembly: dict, *, preliminary: bool = False) -> dict:
    """What a customer needs at a glance: documents with their fields, and what else the video showed."""
    documents = []
    for record in assembly["documents"]:
        kind = record["document_type"]
        labels = FIELD_LABELS[kind]
        fields = []
        for name, label in labels.items():
            field = record["fields"][name]
            if name in {NUMBER_FIELD[kind], DATE_FIELD[kind]} and field["status"] == "recognized":
                continue  # already in the title
            note = STATUS_TEXT.get(field["status"], field["status"])
            if field["status"] == "conflict" and field.get("candidates"):
                note += ": " + " / ".join(candidate.get("raw_value") or BLANK_TEXT for candidate in field["candidates"])
            fields.append({"label": label, "status": field["status"],
                           "value": field["raw_value"] if field["status"] == "recognized" else None,
                           "note": None if field["status"] == "recognized" else note})
        documents.append({"id": record["document_id"], "type": kind, "type_label": record["document_type_label"],
                          "title": document_title(record), "complete": record["status"] == "complete",
                          "undetermined": len(record["undetermined_fields"]), "fields": fields,
                          "sheets": len(record["sheets"]), "start_s": record.get("start_s")})
    forms: "OrderedDict[str, dict]" = OrderedDict()
    for sheet in assembly.get("other_forms", []):
        name = _short_form(sheet.get("observed_form")) or "Форма не определена"
        entry = forms.setdefault(name.casefold(), {"form": name, "sheets": 0,
                                                   "out_of_scope": sheet.get("reason") == "out_of_scope_form_confirmed"})
        entry["sheets"] += 1
    summary = assembly.get("summary", {})
    return {"version": assembly.get("version"), "preliminary": preliminary, "documents": documents,
            "other_forms": list(forms.values()), "other_form_sheets": summary.get("other_form_sheets", 0),
            "unassigned_sheets": summary.get("unassigned_sheets", 0), "sheets_total": summary.get("sheets_total", 0)}
