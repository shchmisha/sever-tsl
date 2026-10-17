#!/usr/bin/env python3
"""Build a CSV worksheet and read-only HTML evidence table from provisional GT.

No model results are read and no provider is called. --embed-images creates a
self-contained HTML with JPEG previews made in memory; source pixels stay intact.
The blank reviewer columns in CSV are for the customer's first review; generating
these files never changes customer_approved or the ground-truth source file.
"""
from __future__ import annotations

import argparse
import base64
import csv
from functools import lru_cache
import hashlib
import html
import io
import json
import os
from pathlib import Path
from urllib.parse import quote


ROLES = {
    "invoice_number": "Номер счёт-фактуры",
    "invoice_date": "Дата счёт-фактуры",
    "document_number": "Номер накладной",
    "document_date": "Дата составления накладной",
    "consignor": "Грузоотправитель — юридическое лицо",
    "consignor_address": "Адрес грузоотправителя",
    "consignee": "Грузополучатель — юридическое лицо",
    "consignee_address": "Адрес грузополучателя",
    "supplier": "Поставщик — юридическое лицо",
    "customer": "Заказчик перевозки — юридическое лицо",
    "document_type": "Тип видимого листа",
}
TYPES = {"upd": "УПД", "ttn": "ТТН", "torg12": "ТОРГ-12", "trn": "ТрН", "unknown": "Не входит в четыре целевых типа"}
VISIBILITY = {
    "visible_readable": "Видно и прочитано",
    "not_visible_on_this_leaf": "Поле этой роли не видно на данной стороне",
    "semantic_confirmation_needed": "Виден иной заголовок: требуется согласование значения поля",
}
CSV_COLUMNS = [
    "record_kind", "source_id", "source_file", "source_sha256", "page_id",
    "start_s", "end_s", "document_type", "observed_form", "field_name",
    "field_role", "labeled_for_value_evaluation", "status", "provisional_value",
    "printed_raw_value", "manual_visibility", "reason", "evidence_frame_id",
    "evidence_timestamp_s", "evidence_source_frame", "evidence_frame_sha256",
    "evidence_upright_view", "reviewer_name", "review_date", "review_status",
    "corrected_value", "comments",
]
MAX_PORTABLE_BYTES = 20_000_000


def esc(value: object) -> str:
    return html.escape(str(value) if value is not None else "", quote=True)


def evidence(video: dict, page: dict, field: dict | None = None) -> dict:
    reviewed = page.get("independent_field_review", {}).get("frames", [])
    if reviewed:
        return reviewed[0]
    provenance = (field or {}).get("provenance")
    if not isinstance(provenance, dict):
        candidates = page.get("provenance", [])
        if isinstance(candidates, str):
            reference = Path(candidates)
            if not reference.is_absolute():
                reference = Path(__file__).resolve().parents[1] / reference
            if reference.is_file():
                observations = json.loads(reference.read_text(encoding="utf-8")).get("observations", [])
                page_number = int(page["page_id"].split("-")[-1])
                observation = next((x for x in observations if x.get("page_candidate_id") == page_number), {})
                selected = observation.get("selected_frame", {})
                if selected.get("path"):
                    return {"source_frame": selected["path"], "upright_view": None,
                            "timestamp_s": selected.get("timestamp_s"),
                            "frame_id": selected.get("sample_id", ""), "source_frame_sha256": ""}
                observation = next((x for x in observations if
                    abs((x.get("approximate_interval_s") or {}).get("start", -1) - page["start_s"]) < 1e-6 and
                    abs((x.get("approximate_interval_s") or {}).get("end", -1) - page["end_s"]) < 1e-6), {})
                references = observation.get("evidence", [])
                if references:
                    selected = references[0]
                    return {"source_frame": selected.get("original_frame"),
                            "upright_view": selected.get("upright_diagnostic_frame"),
                            "timestamp_s": selected.get("timestamp_s"),
                            "frame_id": selected.get("sample_id", ""), "source_frame_sha256": ""}
            provenance = {}
        else:
            provenance = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    if not provenance:
        provenance = next((f.get("provenance", {}) for f in page.get("fields", {}).values() if isinstance(f.get("provenance"), dict)), {})
    frame = provenance.get("original_frame") or provenance.get("frame")
    return {
        "source_frame": frame,
        "upright_view": provenance.get("upright_frame") or provenance.get("crop"),
        "timestamp_s": provenance.get("timestamp_s"),
        "frame_id": provenance.get("sample_id", ""),
        "source_frame_sha256": "",
    }


def worksheet_rows(truth: dict) -> list[dict]:
    rows = []
    for video in truth["videos"]:
        for page in video["pages"]:
            base = {
                "source_id": video["source_id"], "source_file": video["source"],
                "source_sha256": video.get("source_sha256", ""), "page_id": page["page_id"],
                "start_s": page["start_s"], "end_s": page["end_s"],
                "document_type": page["document_type"], "observed_form": page.get("observed_form", ""),
                "reviewer_name": "", "review_date": "", "review_status": "", "corrected_value": "", "comments": "",
            }
            entries = [("leaf_type", "document_type", {
                "value": page["document_type"], "status": "provisional_manual_type",
                "labeled": False, "raw_value": page.get("observed_form", ""),
                "manual_visibility": "type_annotation_for_separate_customer_review",
                "uncertainty": page.get("classification_basis", "Тип предварительный; границы/единицу листа проверяет заказчик."),
            })]
            entries.extend(("required_field", name, page.get("fields", {}).get(name, {
                "value": None, "status": "unreadable", "labeled": False,
                "uncertainty": "Поле ещё не размечено вручную.", "manual_visibility": "unlabelled",
            })) for name in truth["field_contract"].get(page["document_type"], []))
            for kind, name, field in entries:
                ev = evidence(video, page, field)
                rows.append({**base,
                    "record_kind": kind, "field_name": name, "field_role": ROLES[name],
                    "labeled_for_value_evaluation": bool(field.get("labeled", False)) if kind == "required_field" else "not_applicable",
                    "status": field.get("status", ""), "provisional_value": field.get("value") if field.get("value") is not None else "",
                    "printed_raw_value": field.get("raw_value") or "", "manual_visibility": field.get("manual_visibility", "visible_readable" if field.get("labeled") else "unlabelled"),
                    "reason": field.get("uncertainty", ""), "evidence_frame_id": ev.get("frame_id", ""),
                    "evidence_timestamp_s": ev.get("timestamp_s", ""), "evidence_source_frame": ev.get("source_frame") or "",
                    "evidence_frame_sha256": ev.get("source_frame_sha256", ""), "evidence_upright_view": ev.get("upright_view") or "",
                })
    return rows


def local_url(path: str | Path, output_dir: Path) -> str:
    return quote(os.path.relpath(Path(path).resolve(), output_dir.resolve()), safe="/.")


def link(path: str | None, output_dir: Path, label: str) -> str:
    if not path:
        return ""
    if not Path(path).is_file():
        return f'<span class="missing">{esc(label)}: файл не найден ({esc(path)})</span>'
    return f'<a href="{esc(local_url(path, output_dir))}">{esc(label)}</a>'


@lru_cache(maxsize=192)
def pixel_file_sha256(path: str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def document_link(path: Path, output_dir: Path, label: str, *, embedded: bool) -> str:
    if not embedded:
        return link(str(path), output_dir, label)
    if not path.is_file():
        return f'<span class="missing">{esc(label)}: отдельный файл не найден</span>'
    mime = "text/csv" if path.suffix == ".csv" else "application/json"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<a download="{esc(path.name)}" href="data:{mime};base64,{encoded}">{esc(label)} (встроенная копия)</a>'


@lru_cache(maxsize=192)
def jpeg_preview(path: Path, *, target: bool, negative_edge: int = 1600, target_edge: int = 2400) -> tuple[str, dict]:
    from PIL import Image
    limit, quality = (target_edge, 90) if target else (negative_edge, 85)
    # Pixels are used as stored. Existing upright derivatives already encode
    # their documented rotation; no guessed EXIF, new rotation or warp is added.
    with Image.open(path) as original:
        original_size = list(original.size)
        preview = original.convert("RGB")
        preview.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        dimensions = list(preview.size)
        stream = io.BytesIO()
        preview.save(stream, format="JPEG", quality=quality, subsampling=2, optimize=True)
        data = stream.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"), {
        "input_path": str(path), "input_sha256": pixel_file_sha256(str(path)),
        "input_pixel_size": original_size, "preview_pixel_size": dimensions,
        "maximum_long_edge": limit, "jpeg_quality": quality,
        "jpeg_chroma_subsampling": "4:2:0",
        "jpeg_bytes": len(data), "jpeg_sha256": hashlib.sha256(data).hexdigest(),
        "new_rotation_degrees": 0, "new_perspective_transform": False,
    }


def image_block(video: dict, page: dict, output_dir: Path, *, embed_images: bool = False, preview_metadata: list[dict] | None = None, negative_edge: int = 1600, target_edge: int = 2400) -> str:
    ev = evidence(video, page)
    source = ev.get("source_frame")
    view = ev.get("upright_view") or source
    if not view or not Path(view).is_file():
        if embed_images:
            raise ValueError(f"Cannot embed missing proof image: {page['page_id']}")
        return '<p class="missing">Отдельное изображение листа не найдено; используйте исходный контактный лист/видео.</p>'
    # The retained portrait PNG for sample-05-page-005 has no existing oriented
    # derivative. Browser CSS rotates only its display, without writing pixels.
    portrait = not embed_images and not ev.get("upright_view") and ev.get("source_pixel_size") == [2160, 3840]
    klass = "image-viewport portrait" if portrait else "image-viewport"
    ts = ev.get("timestamp_s")
    timing = f"{float(ts):.6f} с исходного видео" if ts is not None else "время не указано в исходной разметке"
    if embed_images:
        image_url, metadata = jpeg_preview(Path(view), target=page["document_type"] != "unknown", negative_edge=negative_edge, target_edge=target_edge)
        metadata = dict(metadata)
        metadata.update({"page_id": page["page_id"], "source_frame_path": source,
                         "source_timestamp_s": ts, "source_video_sha256": video.get("source_sha256")})
        source_hash = pixel_file_sha256(str(source)) if source and Path(source).is_file() else None
        declared_hash = ev.get("source_frame_sha256")
        if declared_hash and source_hash != declared_hash:
            raise ValueError(f"Source image hash differs from declared proof: {page['page_id']}")
        metadata["source_frame_sha256"] = source_hash
        if preview_metadata is not None:
            preview_metadata.append(metadata)
        before = "×".join(str(v) for v in metadata["input_pixel_size"])
        after = "×".join(str(v) for v in metadata["preview_pixel_size"])
        return f'''<figure><div class="{klass}"><img loading="lazy" src="{image_url}" alt="Встроенное фактическое доказательство {esc(page['page_id'])}"></div>
<figcaption>{esc(ev.get('frame_id', ''))}, {esc(timing)}. Встроенное превью JPEG: {before} → {after}, quality={metadata['jpeg_quality']}, RGB, chroma 4:2:0, LANCZOS при уменьшении. Новый поворот и исправление перспективы не применялись.
<small>Исходный кадр: {esc(source)}<br>SHA-256 исходного кадра: {esc(source_hash or 'не доступен')}<br>Файл показанного вида: {esc(view)}<br>SHA-256 показанного вида до уменьшения: {metadata['input_sha256']}<br>SHA-256 JPEG-превью: {metadata['jpeg_sha256']}</small></figcaption></figure>'''
    return f'''<figure><div class="{klass}"><a href="{esc(local_url(view, output_dir))}"><img loading="lazy" src="{esc(local_url(view, output_dir))}" alt="Исходное визуальное доказательство {esc(page['page_id'])}"></a></div>
<figcaption>{esc(ev.get('frame_id', ''))}, {esc(timing)}. {link(source, output_dir, 'Исходные пиксели')} · {link(ev.get('upright_view'), output_dir, 'Развёрнутый вид')}
<small>SHA-256 кадра: {esc(ev.get('source_frame_sha256') or 'сохранённая прежняя разметка; дополнительный хеш не добавлялся')}</small></figcaption></figure>'''


def render_html(truth: dict, rows: list[dict], output: Path, csv_path: Path, source: Path, *, embed_images: bool = False, preview_metadata: list[dict] | None = None, negative_edge: int = 1600, target_edge: int = 2400) -> str:
    output_dir = output.parent
    counts = truth.get("expansion", {}).get("counts", {})
    row_map = {}
    for row in rows:
        row_map.setdefault(row["page_id"], []).append(row)
    parts = []
    for video in truth["videos"]:
        leaves = []
        for page in video["pages"]:
            pid = page["page_id"]
            field_rows = [r for r in row_map[pid] if r["record_kind"] == "required_field"]
            labeled = sum(bool(r["labeled_for_value_evaluation"]) for r in field_rows)
            body = []
            for r in field_rows:
                status = "РАЗМЕЧЕНО" if r["labeled_for_value_evaluation"] else "НЕ РАЗМЕЧЕНО — из точности значений исключить"
                value = esc(r["provisional_value"]) if r["labeled_for_value_evaluation"] else "—"
                body.append(f'''<tr class="{'labelled' if r['labeled_for_value_evaluation'] else 'unlabelled'}"><td>{esc(r['field_role'])}<small>{esc(r['field_name'])}</small></td><td>{value}<small>Печатный литерал: {esc(r['printed_raw_value']) or '—'}</small></td><td>{esc(status)}<small>{esc(VISIBILITY.get(r['manual_visibility'], r['manual_visibility']))}</small></td><td>{esc(r['reason'])}</td><td></td><td></td><td></td><td></td><td></td></tr>''')
            review = page.get("independent_field_review", {})
            extras = []
            for key in ("boundary_note", "occlusion_note", "customer_question"):
                if review.get(key):
                    extras.append(f'<p class="note">{esc(review[key])}</p>')
            observed = review.get("supplemental_observations")
            if observed:
                extras.append(f'<p class="note">Дополнительное печатное наблюдение: собственная шапка «Универсальный передаточный документ № {esc(observed["own_upd_heading_number"])} от {esc(observed["own_upd_heading_date_raw"])}». Это пока не ответ для полей счёт-фактуры.</p>')
            table = f'''<div class="table-scroll"><table><thead><tr><th>Роль</th><th>Предварительное значение / литерал</th><th>Видимость и область оценки</th><th>Причина / ограничения</th><th>Сотрудник</th><th>Дата</th><th>Статус проверки</th><th>Исправленное значение</th><th>Комментарий</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>''' if body else '<p>Для класса unknown обязательные поля четырёх целевых типов не размечаются. Требуется проверка типа листа по изображению.</p>'
            heading = f'{pid} · {TYPES[page["document_type"]]} · {page["start_s"]:.3f}–{page["end_s"]:.3f} с'
            opening = " open" if field_rows else ""
            leaves.append(f'''<details id="{esc(pid)}"{opening}><summary>{esc(heading)} <span>{labeled}/{len(field_rows)} значений размечено</span></summary><p>{esc(page.get('observed_form',''))}</p>{image_block(video,page,output_dir,embed_images=embed_images,preview_metadata=preview_metadata,negative_edge=negative_edge,target_edge=target_edge)}{''.join(extras)}{table}<p class="review-blank">Проверка типа/границ: сотрудник ______ дата ______ статус ______ исправление ______ комментарий ______</p></details>''')
        video_reference = f"Исходное видео (не встроено): {esc(video['source'])}" if embed_images else link(video['source'],output_dir,'Исходное видео')
        parts.append(f'''<section><h2>{esc(video['source_id'])}: {esc(Path(video['source']).name)}</h2><p>{video_reference} · {len(video['pages'])} исходных интервалов; partition={esc(video.get('partition',''))}. <small>SHA-256 видео: {esc(video.get('source_sha256') or 'не указан на уровне видео в исходном JSON')}</small></p>{''.join(leaves)}</section>''')
    questions = ''.join(f'<li>{esc(q)}</li>' for q in truth.get('expansion',{}).get('customer_questions',[]))
    gt_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    csv_link = document_link(csv_path, output_dir, 'CSV таблицу', embedded=embed_images)
    json_link = document_link(source, output_dir, 'Исходный предварительный JSON', embedded=embed_images)
    leaf_count = sum(len(v['pages']) for v in truth['videos'])
    portable_note = f'<p class="note">Переносимая копия: все {leaf_count} превью встроены в этот файл и открываются без рабочей папки. Это реальные сохранённые изображения с явно указанным уменьшением/JPEG-сжатием, не дорисованные данные. Целевые листы: длинная сторона до {target_edge} px, quality 90; нецелевые: до {negative_edge} px, quality 85. JPEG chroma 4:2:0. Для соблюдения размера файла сначала уменьшаются нецелевые превью, затем при необходимости целевые до 2200 px; качество JPEG не снижается. Исходные видео не встроены; их пути/хеши сохранены для сверки. CSV и JSON можно скачать из встроенных копий.</p>' if embed_images else ''
    reproduction = 'python tools/build_review_table.py --embed-images' if embed_images else 'python tools/build_review_table.py'
    image_mode = 'JPEG-превью встроены в файл; исходные изображения и CSV не перезаписывались.' if embed_images else 'изображения используются на месте.'
    return f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Север ТСЛ — предварительный эталон для проверки</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;color:#172331;background:#f5f7fa;margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}h1{{font-size:28px}}h2{{font-size:21px}}p{{max-width:1100px}}a{{color:#146299}}.warning,.note{{padding:12px;background:#fff1ce;border-left:4px solid #c68b20}}details{{background:white;border:1px solid #d7dfe7;border-radius:8px;margin:12px 0;padding:12px}}summary{{cursor:pointer;font-weight:600}}summary span{{float:right;color:#627482}}small{{display:block;font-size:12px;color:#53677a;word-break:break-word}}figure{{margin:12px 0;max-width:1000px}}.image-viewport img{{display:block;max-width:100%;max-height:560px;object-fit:contain;background:#e7ebee}}.portrait{{position:relative;aspect-ratio:16/9;overflow:hidden;width:100%}}.portrait img{{position:absolute;left:50%;top:50%;width:56.25%;max-height:none;max-width:none;transform:translate(-50%,-50%) rotate(-90deg)}}figcaption{{padding-top:6px}}table{{border-collapse:collapse;width:100%;min-width:1200px;font-size:13px}}th,td{{border:1px solid #d2dbe2;padding:8px;vertical-align:top}}th{{text-align:left;background:#e6edf3}}td:empty{{min-width:75px;min-height:30px}}.unlabelled{{background:#fff6e2}}.table-scroll{{overflow:auto}}.review-blank{{font-size:13px;color:#53677a}}.missing{{color:#9a3f23}}nav a{{margin-right:12px}}footer{{margin-top:24px;color:#53677a;font-size:12px}}@media print{{body{{background:white}}details{{break-inside:avoid}}.image-viewport img{{max-height:330px}}table{{min-width:0;font-size:10px}}}}</style></head><body><main>
<h1>Предварительный эталон — проверка сотрудником «Север ТСЛ»</h1>
<p class="warning"><strong>customer_approved=false.</strong> Сотрудник проверяет эталон первым. Это таблица независимых ручных наблюдений, а не результаты модели и не акт приёмки. Значения заранее не передаются модели или frontend как ответы.</p>
{portable_note}
<p>Сохранены {counts.get('visible_leaf_intervals','?')} исходных интервалов, их типы, границы и прежние четыре полных набора полей. После расширения размечено <strong>{counts.get('labelled_field_instances','?')} из {counts.get('required_field_instances','?')}</strong> обязательных экземпляров полей; полностью размечены {counts.get('fully_labelled_target_intervals','?')} из {counts.get('target_leaf_intervals','?')} целевых интервалов. {counts.get('unlabelled_field_instances','?')} неразмеченных значений нельзя засчитывать как правильно распознанные null. Это покрытие разметки, не точность обработки.</p>
<p>Колонки сотрудник/дата/статус/исправление/комментарий пусты. Для внесения результатов проверки используйте {csv_link} в табличном редакторе; HTML содержит только изображения и ссылки, без форм, скриптов или сетевых запросов.</p>
<nav>{json_link} · {csv_link}</nav><h2>Решения заказчика, влияющие на единицу оценки</h2><ul>{questions}</ul>
{''.join(parts)}<footer>Источник JSON SHA-256: {gt_sha}. Перестроение: <code>{reproduction}</code>. Генератор не меняет значения и признак утверждения эталона; {image_mode}</footer></main></body></html>'''


def build(source: Path, csv_path: Path, html_path: Path, *, embed_images: bool = False) -> dict:
    truth = json.loads(source.read_text(encoding="utf-8"))
    if truth.get("customer_approved") is not False:
        raise ValueError("This builder is for provisional, unapproved ground truth only")
    rows = worksheet_rows(truth)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    if not embed_images:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    previews = []
    sizes = [(2400, n) for n in (1600, 1400, 1200, 1000, 800)] + [(2200, n) for n in (1400, 1200, 1000, 800)]
    for target_edge, negative_edge in sizes:
        previews = []
        output = render_html(truth, rows, html_path, csv_path, source, embed_images=embed_images, preview_metadata=previews, negative_edge=negative_edge, target_edge=target_edge)
        output_size = len(output.encode("utf-8"))
        if not embed_images or output_size <= MAX_PORTABLE_BYTES:
            break
    else:
        raise ValueError(f"Portable HTML would be {output_size} bytes; exceeds 20 MB even after reducing negative previews; no output was written")
    html_path.write_text(output, encoding="utf-8")
    return {"customer_approved": False, "source": str(source.resolve()), "csv": str(csv_path.resolve()), "html": str(html_path.resolve()), "rows": len(rows), "leaf_type_rows": sum(r["record_kind"] == "leaf_type" for r in rows), "required_field_rows": sum(r["record_kind"] == "required_field" for r in rows), "labelled_field_rows": sum(r["record_kind"] == "required_field" and r["labeled_for_value_evaluation"] is True for r in rows),
            "embedded_images": embed_images, "embedded_preview_count": len(previews), "embedded_jpeg_bytes": sum(p['jpeg_bytes'] for p in previews), "html_bytes": output_size,
            "preview_settings": {"targets": {"max_long_edge": target_edge, "jpeg_quality": 90}, "unknown": {"max_long_edge": negative_edge, "jpeg_quality": 85}, "jpeg_chroma_subsampling": "4:2:0", "new_pixel_rotation": False, "new_perspective_transform": False},
            "source_and_csv_unchanged": embed_images}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=root / "analysis/ground-truth-expanded-provisional.json")
    parser.add_argument("--csv", type=Path, default=root / "analysis/ground-truth-expanded-review.csv")
    parser.add_argument("--html", type=Path, help="Output HTML; portable default is chosen with --embed-images")
    parser.add_argument("--embed-images", action="store_true", help="Embed in-memory JPEG proof previews and copies of CSV/JSON; keep existing CSV and local HTML")
    args = parser.parse_args()
    html_path = args.html or root / ('analysis/ground-truth-expanded-review-portable.html' if args.embed_images else 'analysis/ground-truth-expanded-review.html')
    print(json.dumps(build(args.ground_truth, args.csv, html_path, embed_images=args.embed_images), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
