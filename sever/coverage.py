"""Path-free coverage disclosures; excluded views are not known missing documents."""
import json
import math
from pathlib import Path
import re


REASONS = {
    "no_stable_brief_view_confirmed": "Короткий показ не подтверждён соседними кадрами",
    "partial_page_overlap_or_insufficient_visible_printed_detail": "Перекрытие листов или недостаточно видимого печатного текста",
    "unconfirmed_excluded_view": "Исключённый кадр требует проверки",
}


def summarize_video_coverage(analysis: dict) -> dict:
    """Summarize sampling exclusions without claiming full coverage or field accuracy."""
    analysis = analysis if isinstance(analysis, dict) else {}
    sampling = analysis.get("sampling")
    sampling = sampling if isinstance(sampling, dict) else {}
    evidence = sampling.get("discarded_transition_evidence")
    evidence = evidence if isinstance(evidence, list) else []
    views, indices = [], set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        index = item.get("sample_index")
        if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
            if index in indices:
                continue
            indices.add(index)
        timestamp = item.get("timestamp_s")
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0):
            timestamp = None
        reason = item.get("reason")
        reason = reason if isinstance(reason, str) and reason in REASONS else "unconfirmed_excluded_view"
        views.append({"timestamp_s": timestamp, "reason": reason})
    discarded = sampling.get("discarded_transition_samples")
    if isinstance(discarded, list):
        for index in discarded:
            if isinstance(index, int) and not isinstance(index, bool) and index >= 0 and index not in indices:
                indices.add(index)
                views.append({"timestamp_s": None, "reason": "unconfirmed_excluded_view"})
    views.sort(key=lambda item: (item["timestamp_s"] is None, item["timestamp_s"] or 0))
    limit = None
    pages = analysis.get("pages")
    for page in pages if isinstance(pages, list) else []:
        segmentation = page.get("segmentation") if isinstance(page, dict) else None
        limitations = segmentation.get("limitations") if isinstance(segmentation, dict) else None
        for item in limitations if isinstance(limitations, list) else []:
            match = re.fullmatch(r"page_limit_reached:(\d+)_candidates_found:(\d+)_retained", item) if isinstance(item, str) else None
            if match:
                found, retained = map(int, match.groups())
                if found > retained and (limit is None or found - retained > limit["unprocessed"]):
                    limit = {"found": found, "retained": retained, "unprocessed": found - retained}
    warnings = []
    if limit:
        warnings.append(f'Неполный охват видео: найдено кандидатов листов — {limit["found"]}, '
                        f'в обработку взяты первые {limit["retained"]}; ещё {limit["unprocessed"]} не обрабатывались. '
                        'Разделите видео на части и загрузите необработанную часть отдельно.')
    if views:
        times = [f'{view["timestamp_s"]:.2f} с' for view in views if view["timestamp_s"] is not None]
        time_text = " Моменты: " + ", ".join(times[:10]) + (" и другие." if len(times) > 10 else ".") if times else ""
        warnings.append(f'Охват требует проверки: исключены неподтверждённые кадры — {len(views)}.' + time_text +
                        ' Это могут быть перелистывания или короткие показы листов; число исключений не равно числу пропущенных документов. '
                        'Проверьте эти моменты в исходном видео.')
    return {"requires_review": bool(warnings), "uncertain_view_count": len(views), "uncertain_views": views,
            "candidate_limit": limit, "warning": " ".join(warnings) or None}


def load_video_coverage(path: Path) -> dict:
    try:
        return summarize_video_coverage(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return summarize_video_coverage({})
