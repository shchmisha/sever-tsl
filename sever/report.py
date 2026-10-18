"""Portable read-only demonstration report for a completed run."""
import html
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from PIL import Image

from .coverage import REASONS, load_video_coverage
from .documents import BLANK_TEXT, COLUMNS, FIELD_COLUMN, TYPE_LABELS, assemble_documents, undetermined_text


_PREVIEW_SIZE = (1024, 760)
_PREVIEW_QUALITY = 82


def _review_image(path, report_root: Path, expected_sha256=None):
    """Only expose existing image evidence within this report's run directory."""
    if not isinstance(path, (str, Path)):
        return None
    try:
        candidate = Path(path).resolve()
        if not candidate.is_relative_to(report_root) or not candidate.is_file():
            return None
        if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            return None
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected_sha256:
                return None
        with Image.open(candidate) as image:
            image.verify()
        return candidate
    except (OSError, ValueError):
        return None


def _review_reasons(adjudication):
    """Retain model explanations without rendering provider requests or metadata."""
    reasons = []
    if isinstance(adjudication, str):
        return [adjudication]
    if not isinstance(adjudication, dict):
        return reasons
    for key in ("reason", "explanation", "summary", "decision_reason", "document_type_reason", "resolution_reason"):
        value = adjudication.get(key)
        if isinstance(value, str) and value:
            reasons.append(value)
    values = adjudication.get("reasons")
    if isinstance(values, list):
        reasons.extend(value for value in values if isinstance(value, str) and value)
    decision = adjudication.get("decision")
    if isinstance(decision, dict):
        reasons.extend(_review_reasons(decision))
    model_review = adjudication.get("model_review")
    if isinstance(model_review, dict):
        reasons.extend(_review_reasons(model_review))
    field_reviews = adjudication.get("field_reviews")
    if isinstance(field_reviews, dict):
        for name, review in field_reviews.items():
            model_review = review.get("model_review") if isinstance(review, dict) else None
            if isinstance(model_review, dict):
                reasons.extend(f"{name}: {reason}" for reason in _review_reasons(model_review))
    return list(dict.fromkeys(reasons))


def _conflict_review_sections(reviews, source_frames, report_root, esc, url):
    sections, public_reviews = [], []
    verified_sources = {identifier: _review_image(path, report_root) for identifier, path in source_frames.items()}
    source_digests = {}
    def source_digest(identifier):
        if identifier not in source_digests:
            try:
                source_digests[identifier] = hashlib.sha256(verified_sources[identifier].read_bytes()).hexdigest()
            except (OSError, AttributeError, KeyError):
                source_digests[identifier] = None
        return source_digests[identifier]
    for review in reviews:
        if not isinstance(review, dict):
            continue
        answer = review.get("answer")
        answer = answer if isinstance(answer, dict) else {}
        phase = review.get("phase")
        status = review.get("status")
        type_review = review.get("kind") == "document_type"
        phase_label = {"initial": "Первичная повторная проверка", "final": "Итоговая повторная проверка"}.get(phase, "Повторная проверка")
        status_label = {"resolved": "Разногласия разрешены", "unresolved": "Разногласия сохранились", "failed": "Повторная проверка не завершена", "processing": "Повторная проверка продолжается"}.get(status, "Статус проверки не указан")
        invalidated = review.get("invalidated_by")
        invalidated = invalidated if isinstance(invalidated, dict) else {}
        if invalidated:
            status_label = "Решение проверки оспорено новым чтением"
        frame_ids = review.get("frame_ids") or []
        frame_ids = [identifier for identifier in frame_ids if isinstance(identifier, str)] if isinstance(frame_ids, list) else []
        def frame_link(identifier):
            source = verified_sources.get(identifier) if identifier in frame_ids else None
            return f'<a href="{url(source)}">{esc(identifier)}</a>' if source else esc(identifier)
        recognition = answer.get("recognition") or {}
        proof = recognition.get("pixel_proof") or {} if isinstance(recognition, dict) else {}
        sources = proof.get("sources") or {} if isinstance(proof, dict) else {}
        frame_labels = {}
        if isinstance(sources, dict):
            for label, source in sources.items():
                if not isinstance(source, dict):
                    continue
                identifier = source.get("frame_id")
                if (isinstance(identifier, str) and identifier in frame_ids and verified_sources.get(identifier)
                        and source.get("source_sha256") == source_digest(identifier)):
                    frame_labels[label] = identifier
        frames = ", ".join(frame_link(identifier) for identifier in frame_ids)
        fields = answer.get("fields")
        fields = fields if isinstance(fields, dict) else {}
        rows, public_fields, joint_tables, public_joint = [], {}, [], {}
        adjudication = answer.get("adjudication") or {}
        field_reviews = adjudication.get("field_reviews") or {} if isinstance(adjudication, dict) else {}
        for name, field in fields.items():
            if not isinstance(field, dict):
                continue
            evidence_rows, public_evidence = [], []
            for evidence in field.get("evidence") or []:
                if not isinstance(evidence, dict):
                    continue
                identifier = evidence.get("frame_id", "")
                identifier = identifier if isinstance(identifier, str) else ""
                evidence_rows.append(f'{frame_link(identifier)}: {esc(evidence.get("text"))}')
                public_evidence.append({"frame_id": identifier, "text": evidence.get("text")})
            rows.append(f'<tr><td>{esc(name)}</td><td>{esc(field.get("raw_value"))}</td><td>{esc(field.get("status"))}</td><td>{esc(field.get("reason"))}</td><td>{"<br>".join(evidence_rows)}</td></tr>')
            public_fields[name] = {key: field.get(key) for key in ("raw_value", "status", "reason")}
            public_fields[name]["evidence"] = public_evidence
            field_review = field_reviews.get(name) or {} if isinstance(field_reviews, dict) else {}
            model_review = field_review.get("model_review") or {} if isinstance(field_review, dict) else {}
            readings = model_review.get("readings") or [] if isinstance(model_review, dict) else []
            joint_rows, compact_readings = [], []
            for reading in readings if isinstance(readings, list) else []:
                if not isinstance(reading, dict):
                    continue
                label = reading.get("frame_id")
                label = label if isinstance(label, str) else ""
                actual_id = frame_labels.get(label)
                reference = frame_link(actual_id) if actual_id else esc(label) + " (неподтверждённая метка)"
                joint_rows.append(f'<tr><td>{reference}</td><td>{esc(reading.get("raw_value"))}</td><td>{esc(reading.get("status"))}</td><td>{esc(reading.get("evidence_text"))}</td></tr>')
                compact_readings.append({"frame_id": actual_id or label, "frame_binding_verified": actual_id is not None,
                    **{key: reading.get(key) for key in ("raw_value", "status", "evidence_text")}})
            if joint_rows:
                joint_tables.append(f'<h5>{esc(name)} · Предложения совместного ответа по каждому кадру</h5>'
                    '<div class="table"><table><thead><tr><th>Кадр</th><th>Буквальное значение</th><th>Статус модели</th><th>Цитата модели</th></tr></thead><tbody>'
                    + "".join(joint_rows) + '</tbody></table></div>')
                public_joint[name] = compact_readings
        table = ('<div class="table"><table><thead><tr><th>Поле</th><th>Ответ модели при повторной проверке</th><th>Статус</th><th>Причина</th><th>Кадры и цитаты</th></tr></thead><tbody>'
                 + "".join(rows) + '</tbody></table></div>') if rows else '<p>Ответ с проверенными полями не сохранён.</p>'
        reasons = _review_reasons(answer.get("adjudication"))
        explanation = '<p>Пояснение модели:</p><ul>' + "".join(f'<li>{esc(reason)}</li>' for reason in reasons) + '</ul>' if reasons else ""
        proof_links, verified_panels = [], {}
        panels = answer.get("panels") or {}
        panels = panels.values() if isinstance(panels, dict) else panels if isinstance(panels, list) else []
        for panel in panels:
            if not isinstance(panel, dict) or panel.get("frame_id") not in frame_ids or not verified_sources.get(panel.get("frame_id")):
                continue
            # A proof link must bind actual saved bytes, never a model-proposed URL.
            identifier = panel["frame_id"]
            if panel.get("source_sha256") != source_digest(identifier):
                continue
            digest = panel.get("upload_sha256")
            path = _review_image(panel.get("upload_path"), report_root, digest) if digest else None
            if path:
                label = f'{panel.get("frame_id")}: {panel.get("panel_id", "фрагмент")}'
                link = f'<a href="{url(path)}">{esc(label)}</a>'
                proof_links.append(link)
                verified_panels[panel.get("panel_id")] = {"frame_id": identifier, "link": link}
        public_type_readings = []
        if type_review:
            table = (f'<p>Ответ совместной проверки по форме: <strong>{esc(answer.get("document_type"))}</strong></p>'
                     if answer.get("document_type") else '<p>Ответ по форме не сохранён.</p>')
            model_review = adjudication.get("model_review") or {} if isinstance(adjudication, dict) else {}
            type_readings = model_review.get("readings") or [] if isinstance(model_review, dict) else []
            type_rows = []
            for reading in type_readings if isinstance(type_readings, list) else []:
                if not isinstance(reading, dict):
                    continue
                label = reading.get("frame_id")
                label = label if isinstance(label, str) else ""
                actual_id = frame_labels.get(label)
                reference = frame_link(actual_id) if actual_id else esc(label) + " (неподтверждённая метка)"
                citations = []
                panel_ids = reading.get("panel_ids") or []
                panel_ids = [identifier for identifier in panel_ids if isinstance(identifier, str)] if isinstance(panel_ids, list) else []
                for identifier in panel_ids:
                    panel = verified_panels.get(identifier)
                    citations.append(panel["link"] if panel and actual_id and panel["frame_id"] == actual_id
                                     else esc(identifier) + " (неподтверждённый фрагмент)")
                type_rows.append(f'<tr><td>{reference}</td><td>{esc(reading.get("document_type"))}</td><td>{esc(reading.get("heading_text"))}</td><td>{"<br>".join(citations)}</td></tr>')
                public_type_readings.append({"frame_id": actual_id or label, "frame_binding_verified": actual_id is not None,
                    **{key: reading.get(key) for key in ("document_type", "heading_text")}, "panel_ids": panel_ids})
            if type_rows:
                joint_tables = ['<h5>Форма документа · Предложения совместного ответа по каждому кадру</h5>'
                    '<div class="table"><table><thead><tr><th>Кадр</th><th>Тип модели</th><th>Буквальный собственный заголовок</th><th>Изображения, указанные моделью</th></tr></thead><tbody>'
                    + "".join(type_rows) + '</tbody></table></div>']
        proofs = '<p>Переданные модели фрагменты: ' + " · ".join(proof_links) + '</p>' if proof_links else ""
        error = f'<p>Ошибка проверки: {esc(review.get("error_type"))}. Исходные разногласия сохранены.</p>' if status == "failed" else ""
        invalidation = ""
        public_invalidation = {}
        if invalidated:
            identifier = invalidated.get("frame_id")
            identifier = identifier if isinstance(identifier, str) else ""
            source = verified_sources.get(identifier)
            reference = f'<a href="{url(source)}">{esc(identifier)}</a>' if source else esc(identifier)
            invalidation = (f'<p>Позднейшее чтение кадра {reference} вернуло другой тип: '
                            f'{esc(invalidated.get("document_type"))}. Предыдущее совместное решение сохранено как история; '
                            f'оно больше не подтверждает текущий результат. {esc(invalidated.get("reason"))}</p>')
            public_invalidation = {key: invalidated.get(key) for key in ("frame_id", "document_type", "reason")}
        field_names = review.get("field_names") or []
        field_names = [name for name in field_names if isinstance(name, str)] if isinstance(field_names, list) else []
        joint = ('<p>Строки ниже — части одного совместного ответа модели по указанным кадрам.</p>'
                 + "".join(joint_tables)) if joint_tables else ""
        subject = "Проверка формы основного документа" if type_review else "Спорные поля: " + esc(", ".join(field_names))
        sections.append(f'<section class="conflict-review"><h4>{phase_label} · {status_label}</h4><p>Кадры: {frames}</p><p>{subject}</p>{error}{invalidation}{explanation}{table}{joint}{proofs}</section>')
        public_answer = {"fields": public_fields, "model_reasons": reasons, "joint_readings": public_joint}
        if type_review:
            public_answer.update(document_type=answer.get("document_type"), joint_type_readings=public_type_readings)
        public_reviews.append({"kind": "document_type" if type_review else "fields", "phase": phase,
                               "status": status, "frame_ids": frame_ids, "field_names": field_names,
                               "error_type": review.get("error_type"), "invalidated_by": public_invalidation,
                               "answer": public_answer})
    rendered = ('<details class="conflict-reviews"><summary>Повторная проверка спорных полей по изображениям</summary>'
                '<p>Это новый совместный ответ модели по нескольким изображениям, отдельный от независимых чтений каждого кадра. '
                'Исходные ответы модели сохранены в таблицах выше.</p>' + "".join(sections) + '</details>') if sections else ""
    return rendered, public_reviews


def _frame_preview(path: Path, assets: Path) -> tuple[Path, tuple[int, int]]:
    """Derive a display image, keeping recognition/source pixels untouched."""
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        target = assets / f"{digest.hexdigest()}-1024x760-q82.jpg"
        if target.is_file():
            try:
                with Image.open(target) as cached:
                    if cached.format == "JPEG" and all(0 < n <= bound for n, bound in zip(cached.size, _PREVIEW_SIZE)):
                        return target, cached.size
            except OSError:
                pass  # Regenerate a damaged derived asset from its actual source.
        source.seek(0)
        with Image.open(source) as image:
            image.thumbnail(_PREVIEW_SIZE, Image.Resampling.LANCZOS)
            preview = image.convert("RGB")
        assets.mkdir(parents=True, exist_ok=True)
        temporary = assets / f".{target.stem}-{uuid4().hex}.tmp"
        try:
            preview.save(temporary, format="JPEG", quality=_PREVIEW_QUALITY, optimize=True)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target, preview.size

def _documents_section(assembly: dict, esc, processing: bool) -> str:
    """Customer view first: one row per document, one column per contract field."""
    summary = assembly["summary"]
    by_type = ", ".join(f"{TYPE_LABELS[kind]} {count}" for kind, count in summary["documents_by_type"].items())
    rows = []
    titles = dict(COLUMNS)
    for record in assembly["documents"]:
        cells = {title: "" for _, title in COLUMNS}
        for name, field in record["fields"].items():
            value = field["raw_value"] if field["status"] == "recognized" else BLANK_TEXT if field["status"] == "not_present" else ""
            cells[titles[FIELD_COLUMN[name]]] = value or ""
        sheets = ", ".join(
            f'<a href="#sheet-{esc(sheet["page_id"])}">{sheet["start_s"]:.1f}–{sheet["end_s"]:.1f} с</a>'
            for sheet in record["sheets"]
            if isinstance(sheet.get("start_s"), (int, float)) and isinstance(sheet.get("end_s"), (int, float)))
        missing = undetermined_text(record)
        rows.append("<tr>" + f'<td>{esc(record["document_type_label"])}</td>'
                    + "".join(f"<td>{esc(cells[title])}</td>" for _, title in COLUMNS)
                    + f'<td class="missing">{esc(missing)}</td><td>{sheets}</td></tr>')
    head = "<tr><th>Тип</th>" + "".join(f"<th>{esc(title)}</th>" for _, title in COLUMNS) + "<th>Не удалось определить</th><th>Листы в видео</th></tr>"
    table = (f'<div class="table"><table class="documents"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'
             if rows else "<p>Документов четырёх типов из ТЗ не найдено.</p>")
    note = "Предварительно: обработка продолжается, список документов ещё может измениться. " if processing else ""
    extra = []
    if summary["other_form_sheets"]:
        extra.append(f'листов других форм вне ТЗ: {summary["other_form_sheets"]}')
    if summary["unassigned_sheets"]:
        extra.append(f'листов без номера и без данных, не отнесённых к документу: {summary["unassigned_sheets"]}')
    return (f'<section id="documents"><h2>Документы: {summary["documents"]}' + (f" ({esc(by_type)})" if by_type else "") + "</h2>"
            f'<p>{esc(note)}Листы одного документа (копии, оборот, продолжение, повторный показ) объединены по типу и собственному номеру. '
            f'Каждое значение взято без изменений с листа, где оно напечатано. Всего листов в видео: {summary["sheets_total"]}'
            + (f'; {esc("; ".join(extra))}' if extra else "") + ".</p>" + table + "</section>")


def write_report(out:Path,result:dict,*,report_root:Path|None=None):
    out = Path(out).resolve()
    # A staged HTML export can link to persistent assets in its final run folder.
    report_root = Path(report_root).resolve() if report_root is not None else out
    assets = report_root / "report-assets"
    def esc(value): return html.escape(str(value if value is not None else ""))
    def url(path): return esc(quote(os.path.relpath(path, report_root).replace(os.sep, "/"), safe="/"))
    previews = {}
    cards=[]
    assembly = result.get("documents") if isinstance(result.get("documents"), dict) else assemble_documents(result)
    membership = {sheet["page_id"]: f'{record["document_type_label"]} № {record["number"]}' if record["number"] else f'{record["document_type_label"]} без номера'
                  for record in assembly["documents"] for sheet in record["sheets"]}
    for page in result["pages"]:
        doc=page["result"]
        frames=[]
        for frame in page["frames"]:
            path=Path(frame["path"]).resolve()
            if path not in previews:
                try:
                    previews[path] = _frame_preview(path, assets)
                except OSError:
                    # Preserve evidence links when a source cannot be decoded or
                    # optional preview storage is unavailable. Never silently make
                    # the browser fetch a full-resolution frame instead.
                    previews[path] = None
            preview = previews[path]
            if preview:
                preview_path, (width, height) = preview
                image = f'<img loading="lazy" decoding="async" width="{width}" height="{height}" src="{url(preview_path)}" alt="{esc(frame["frame_id"])}">'
            else:
                image = "Превью недоступно · открыть исходный кадр"
            selected = " · выбранный кадр" if frame["frame_id"] == page.get("selected_frame_id") else ""
            frames.append(f'<figure><a href="{url(path)}" title="Открыть исходный кадр в полном разрешении">{image}</a><figcaption>{esc(frame["frame_id"])} · {frame["timestamp_s"]:.2f} с{selected}</figcaption></figure>')
        rows=[]
        for name,field in doc["fields"].items():
            evidence=" | ".join(f'{e["frame_id"]}: {e["text"]}' for e in field.get("evidence",[]))
            rows.append(f'<tr><td>{esc(name)}</td><td>{esc(field["raw_value"])}</td><td>{esc(field["normalized_value"])}</td><td>{esc(field["status"])}</td><td>{esc(evidence)}</td></tr>')
        comparison=page.get("neighbor_comparison",{})
        views=f'<p>Проверка кадров: {esc(comparison.get("status"))}; рассмотрено дополнительных видов: {len(comparison.get("attempts",[]))}. Неразрешённые или неподтверждённые поля: {esc(", ".join(comparison.get("remaining_fields",[])))}</p>' if comparison else ""
        inn_rows=[]
        for role,check in page.get("inn_verification",{}).get("roles",{}).items():
            registry=check.get("registry",{})
            inn_rows.append(f'<tr><td>{esc(role)}</td><td>{esc(check.get("printed_inn"))}</td><td>{esc(check.get("printed_company_name"))}</td><td>{esc(registry.get("status"))}</td><td>{esc(registry.get("reason"))}</td></tr>')
        inns='<h3>Сверка ИНН с ФНС</h3><table><thead><tr><th>Роль</th><th>Печатный ИНН</th><th>Печатное название</th><th>Результат реестра</th><th>Пояснение</th></tr></thead><tbody>'+"".join(inn_rows)+'</tbody></table><p>Проверяется текущая запись реестра; исходные реквизиты не заменяются.</p>' if inn_rows else ""
        display_type = "Форма не подтверждена" if doc["document_type"] == "unknown" else doc["document_type"]
        native_stop = (page.get("image_recognition") or {}).get("stop_reason")
        if doc["document_type"] == "unknown" and native_stop == "out_of_scope_form_confirmed":
            display_type = "Прочитана другая форма документа"
        fragments = ""
        source_frames = {frame["frame_id"]: Path(frame["path"]).resolve() for frame in page["frames"]}
        if "image_recognition" in page:
            image_readings = []
            for reading in (page.get("image_recognition") or {}).get("views", []):
                identifier = reading.get("frame_id", "")
                source = source_frames.get(identifier)
                if source is None and reading.get("frame_path"):
                    source = Path(reading["frame_path"]).resolve()
                reference = f'<a href="{url(source)}">{esc(identifier)}</a>' if source else esc(identifier)
                observed_form = reading.get("observed_form") or "Форма не прочитана"
                scope = reading.get("form_scope")
                scope_note = ("Форма вне четырёх типов, предусмотренных ТЗ." if scope == "out_of_scope"
                              else "Недостаточно видимых данных для определения формы." if scope == "insufficient_evidence"
                              else "")
                visible_text = reading.get("visible_text") or ""
                text = f'<pre class="image-text">{esc(visible_text)}</pre>' if visible_text.strip() else '<p>Модель не вернула читаемый текст для этого вида.</p>'
                proposals = []
                for name, field in (reading.get("document") or {}).get("fields", {}).items():
                    proposals.append(f'<tr><td>{esc(name)}</td><td>{esc(field.get("raw_value"))}</td><td>{esc(field.get("status"))}</td></tr>')
                proposal_table = ('<div class="table"><table><thead><tr><th>Поле</th><th>Ответ модели на этом кадре</th><th>Статус модели</th></tr></thead><tbody>'
                                  + "".join(proposals) + '</tbody></table></div>') if proposals else ""
                image_readings.append(f'<section class="image-reading"><p>{reference} · Определённая моделью форма: <strong>{esc(observed_form)}</strong></p><p>{esc(scope_note)}</p>{text}{proposal_table}</section>')
            fragments = '<h3>Чтение изображения моделью</h3><p>Текст приведён из ответа модели для указанного кадра. Координаты доказательств относятся к изображению или фрагменту.</p>' + ("".join(image_readings) if image_readings else '<p>В результате нет сохранённого чтения изображения.</p>')
        elif doc["document_type"] == "unknown":
            excerpts = []
            for evidence in doc.get("classification_evidence", []):
                if not evidence.get("text", "").strip():
                    continue
                identifier = evidence.get("frame_id", "")
                reference = f'<a href="{url(source_frames[identifier])}">{esc(identifier)}</a>' if identifier in source_frames else esc(identifier)
                excerpts.append(f'<li>{reference}: {esc(evidence["text"])}</li>')
            fragments = '<h3>Прочитанные фрагменты</h3>' + ('<ul>' + "".join(excerpts) + '</ul>' if excerpts else
                '<p>В результате нет подтверждающих цитат. Исходные кадры доступны по ссылкам ниже.</p>')
        conflict_reviews = ""
        details_page = page
        recognition = page.get("image_recognition") or {}
        if recognition.get("conflict_reviews"):
            # Additional read frames are already bound by the saved independent
            # views; raw review transport is intentionally absent from HTML.
            review_sources = dict(source_frames)
            for reading in recognition.get("views", []):
                if reading.get("frame_id") not in review_sources and reading.get("frame_path"):
                    review_sources[reading["frame_id"]] = Path(reading["frame_path"]).resolve()
            conflict_reviews, public_reviews = _conflict_review_sections(
                recognition["conflict_reviews"], review_sources, report_root, esc, url)
            details_page = {**page, "image_recognition": {**recognition, "conflict_reviews": public_reviews}}
        field_table = '<div class="table"><table><thead><tr><th>Поле</th><th>Исходное значение</th><th>Нормализованное</th><th>Статус</th><th>Доказательство</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>' if rows else '<p>Подтверждённых полей нет.</p>'
        belongs = membership.get(page["page_id"])
        role = f"лист документа {belongs}" if belongs else ("другая форма, не входит в ТЗ" if doc["document_type"] == "unknown" else "не отнесён к документу")
        cards.append(f'<details class="sheet"><summary>{esc(page["page_id"])} · {esc(display_type)} · {page["start_s"]:.2f}–{page["end_s"]:.2f} с · {esc(role)}</summary><article id="sheet-{esc(page["page_id"])}"><h2>{esc(page["page_id"])} · {esc(display_type)} · {page["start_s"]:.2f}–{page["end_s"]:.2f} с</h2><p>{esc("; ".join(doc["errors"]+doc["notes"]))}</p>{views}{fragments}{conflict_reviews}<div class="frames">{"".join(frames)}</div>{field_table}{inns}<details><summary>Качество и координаты доказательств</summary><pre>{esc(json.dumps(details_page,ensure_ascii=False,indent=2))}</pre></details></article></details>')
    processing = result.get("status") == "processing"
    failed = result.get("status") == "failed"
    refresh = '<meta http-equiv="refresh" content="10">' if processing else ""
    if processing:
        title = heading = "Север ТСЛ — обработка продолжается"
        notice = ('<p role="status"><strong>Это промежуточный результат.</strong> '
                  'Показаны уже полученные чтения; обработка продолжается, значения ещё могут уточняться.</p>')
    elif failed:
        title = heading = "Север ТСЛ — Обработка остановлена"
        notice = ('<p role="alert"><strong>Это промежуточный результат.</strong> '
                  'Сохранены чтения, полученные до остановки. Итоговый результат не сформирован.</p>')
    else:
        title = "Север ТСЛ — результат"
        heading = "Север ТСЛ — результат обработки"
        notice = ""
    coverage = result.get("video_coverage") or load_video_coverage(out / "video" / "video-analysis.json")
    if coverage.get("warning"):
        heading_text = "Видео обработано не полностью." if coverage.get("candidate_limit") else "Охват видео требует проверки."
        details = []
        for view in coverage.get("uncertain_views", []):
            timestamp = view.get("timestamp_s")
            moment = f"{timestamp:.2f} с" if isinstance(timestamp, (int, float)) else "Время не сохранено"
            details.append(f'<li>{esc(moment)} — {esc(REASONS.get(view.get("reason"), REASONS["unconfirmed_excluded_view"]))}</li>')
        exclusions = ('<details><summary>Исключённые неподтверждённые кадры</summary><ul>' + "".join(details) + '</ul></details>') if details else ""
        notice += f'<section id="video-coverage-warning" role="alert"><p><strong>{heading_text}</strong> {esc(coverage["warning"])}</p>{exclusions}</section>'
    links = ('<a href="partial-results.json">Промежуточный JSON</a> · <a href="trace.jsonl">Трасса шагов</a>' if processing or failed
             else '<a href="documents.csv">CSV по документам</a> · <a href="results.json">JSON</a> · <a href="fields.csv">CSV по листам</a> · <a href="trace.jsonl">Трасса шагов</a>')
    documents_section = _documents_section(assembly, esc, processing or failed)
    page=f'''<!doctype html><html lang="ru"><meta charset="utf-8">{refresh}<meta name="viewport" content="width=device-width"><title>{title}</title><style>body{{font:15px system-ui;max-width:1300px;margin:28px auto;padding:0 18px;color:#182333;background:#f4f6fa}}article{{background:white;padding:20px;margin:20px 0;border-radius:10px}}h1{{font-size:26px}}h2{{font-size:18px}}.frames{{display:flex;flex-wrap:wrap;gap:20px}}figure{{margin:0}}img{{display:block;width:auto;height:auto;max-height:380px;max-width:42vw}}figcaption{{font-size:12px;color:#586475}}.table{{overflow:auto}}table{{border-collapse:collapse;margin-top:18px;width:100%}}td,th{{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #dae0ea}}pre{{white-space:pre-wrap;font-size:12px}}.image-text{{padding:12px;background:#f4f6fa;font-size:14px;overflow-wrap:anywhere}}a{{color:#165dca}}details.sheet{{background:white;margin:10px 0;border-radius:10px;padding:10px 16px}}details.sheet>summary{{cursor:pointer;font-weight:600}}details.sheet article{{margin:0;padding:10px 0}}#documents{{background:white;padding:20px;margin:20px 0;border-radius:10px}}table.documents td,table.documents th{{font-size:14px}}td.missing{{color:#a33}}</style><h1>{heading}</h1>{notice}<p>Проверяемые формы: УПД, ТОРГ-12, товарно-транспортная накладная и транспортная накладная.</p><p>{esc(result["source"]["path"])}<br>SHA-256: {esc(result["source"]["sha256"])}</p><p>{len(result["pages"])} кандидатов листов · {result["elapsed_seconds"]:.1f} с обработки. Уверенность модели и показатели качества не являются измеренной точностью.</p><p>{links}</p>{documents_section}<h2>Листы видео подробно</h2><p>Распознавание каждого листа с кадрами и ответами модели. Нажмите на строку листа, чтобы раскрыть его.</p>{"".join(cards)}</html>'''
    (out/"report.html").write_text(page,encoding="utf-8")
