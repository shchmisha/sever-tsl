"""Fresh joint pixel review of contested forms/fields, with no previous answers."""
from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_ai import Agent, BinaryContent, ModelRetry, ToolOutput, ToolReturn
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RunUsage, UsageLimits
from openai import AsyncOpenAI

from .agents import (AgentConfig, AnalysisError, BatchBudgetExceeded, FatalProviderError,
                     _PageDeps, _RouterAIChatModel, _after_request, _before_request,
                     _close_run_usage, _dump, _settle_run_usage, _trace, _write_cache as _write_evidence)
from .image_agent import Coordinate, EvidencePersistenceError, _Source
from .image_fusion import _field_comparison
from .image_reader import (EVIDENCE_ORIGIN, TRNFields, TTNFields, TORGFields,
                           UPDFields, _sha)
from .normalize import normalize
from .schema import DocumentType, Evidence, FIELDS_BY_TYPE, FieldValue

IMAGE_CONFLICTS_VERSION = "fresh-joint-source-visibility-and-field-review-v7.1-ttn-continuation"
MAX_MODEL_REQUESTS = 12
MAX_ZOOM_IMAGES = 12
MAX_REVIEW_FRAMES = 4
_FIELD_MODELS = {"upd": UPDFields, "torg12": TORGFields, "ttn": TTNFields, "trn": TRNFields}

INSTRUCTIONS = """
Перечитай спорные поля заново по реальным изображениям одного листа.
Предыдущие ответы и ожидаемые значения тебе не передаются. Проверь каждое
запрошенное поле отдельно на КАЖДОМ предоставленном кадре и сохрани его
буквальное значение, строку доказательства и реальные panel_ids этого кадра.
Не переноси прочитанное на одном кадре в ответ для другого. При нечитаемой
строке верни unreadable, не дополняй её по остальным кадрам или смыслу.
Прочитай собственный основной документ, не соседнюю бумагу и не ссылку на
приложенный документ. Подтверди его форму по читаемым изображениям. Если
видны разные подтверждённые формы, document_type=unknown; не смешивай листы.
Отдельный нечитаемый кадр сохраняй unreadable, не выдавай его за подтверждение
и не придумывай ему другое значение. Два читаемых ракурса могут подтвердить
поле при наличии дополнительных нечитаемых, но никакое другое реально
прочитанное значение нельзя отбросить ради согласия.
Организация — её название в собственной явно подписанной роли, а не ИНН,
банковские реквизиты, оператор электронного обмена или создатель документа
внизу страницы. Плательщик, перевозчик, пункт погрузки/разгрузки и заказчик
не заменяют грузоотправителя или грузополучателя.
Если в собственном блоке роли виден только адрес, а название организации
не читается или не указано, для организационного поля верни unreadable и
raw_value=null. Полный видимый блок сохрани в evidence_text; не подставляй
адрес вместо названия и не заимствуй организацию из другой роли. Это правило
организационных полей не сокращает полное значение адресного поля.
В соседних строках сначала прочитай подпись СЛЕВА и её собственный текст
СПРАВА на той же горизонтальной строке. Не меняй местами отправителя и
получателя. Если непосредственно после подписи ЗАПРОШЕННОЙ роли напечатано
название организации, выпиши его напрямую. Ссылка «он же» из СОСЕДНЕЙ строки
к этой роли не относится. Раскрывай ссылку только когда она буквально
напечатана в собственной строке запрошенного поля; тогда сохрани отдельно
обе настоящие строки. raw_value НИКОГДА не содержит слов «он же», «тот же»
или другой ссылки: это указатель, а не значение. Строку со ссылкой сохрани
в evidence_text, а в raw_value запиши только само значение, на которое она
указывает: наименование продавца либо ПОЛНЫЙ напечатанный адресный блок
продавца, без префикса «он же;». В evidence_text копируй именно видимую подпись и текст
этой строки дословно, включая продолжение адреса. Не склеивай разные строки
с выдуманными подписями, не добавляй [он же], многоточия и пояснения от себя.
resolution_reason кратко объясняет чтение и не должен противоречить этим
буквальным строкам; не утверждай совпадение ролей без напечатанной ссылки.
Адресное поле читай ЦЕЛИКОМ: всё, что действительно напечатано в собственном
поле/блоке адреса нужной роли, включая название организации, РЦ/склада,
ИНН/КПП и другие реквизиты, если они входят именно в этот блок. Не сокращай
его до почтового адреса и не вырезай части по их смыслу. Границы определяй по
подписи, разметке и продолжениям самого поля на изображении; соседний блок
«Покупатель» или другой роли не добавляй. Сохрани полную исходную строку в
evidence_text. Читай дословно, включая сокращения и цифры; не исправляй
названия улиц, почтовые адреса и неоднозначные буквы по вероятности.
Сначала запрашивай нужные участки всех кадров вместе через zoom_regions:
один вызов со списком {frame_id, left, top, right, bottom} сохраняет нативные
пиксели каждого участка и сокращает число последовательных запросов модели.
Для одного дополнительного участка используй zoom_source с теми же полями.
frame_id — предоставленная метка frame_01 и т.д.; координаты [0,1] относятся
к ЦЕЛОМУ ПОВЁРНУТОМУ изображению именно этого кадра, а не к предыдущему crop.
Выбирай реальные нативные участки с подписью роли и её значением. panel_ids
должны ссылаться только на изображения, реально полученные для этого кадра.
Для not_present собственное место поля должно быть явно видимо пустым на
каждом кадре; raw_value=null, explicit_blank=true, complete_page_visible=true.
Вне crop не означает пусто.
Для каждого поля и каждого кадра явно укажи visibility_cause. Если подпись
или текст размыты, малы, закрыты, обрезаны, ещё не увеличены либо форма не
уверенно установлена, это НЕ other_document_side: нужны другие ракурсы или
увеличения. Только когда полная собственная сторона/страница-продолжение
видна, её структура читается и запрошенное поле находится на ДРУГОЙ стороне
документа, выбери other_document_side и next_evidence_action=
obtain_other_document_side. Не делай этот вывод просто по отсутствию чтения,
названию формы или сходству ответов. Подтверди его отдельно настоящими
panel_ids и буквальным видимым контекстом каждого кадра. Значение остаётся
unreadable с raw_value=null; это не пустое поле и не успешное распознавание.
Если по читаемой структуре ПОЛНОГО листа установлено, что САМОГО БЛОКА
или строки на этой стороне нет и нужна другая сторона, это unreadable +
other_document_side, а НЕ not_present. Если блок мог быть не замечен или
неразличим, причина остаётся unknown/blur_or_small_text/needs_more_zoom.
not_present допустим только для реально существующего, явно видимого
и пустого собственного поля.
Для противоречий, временного закрытия и любых сомнений выбери read_more_views.
У тебя не более 12 запросов и 12 изображений увеличения на этот совместный
проход. Последний запрос оставлен для submit_conflict_review. При отказе tools
новое изображение не выдано; используй полученные пиксели или unreadable.
Заверши через submit_conflict_review. Не выбирай большинство: если изображения
показывают разные значения, сохрани отдельные чтения и объясни расхождение.
"""


TYPE_INSTRUCTIONS = """
Прочитай собственную форму документа заново на КАЖДОМ реальном кадре одного
листа. Предыдущие ответы, спорящие типы и ожидаемый результат тебе не переданы.
Для каждого кадра верни отдельно document_type, буквальный собственный
заголовок heading_text и реальные panel_ids именно этого кадра. Не копируй
прочитанное на другом кадре. Если собственного заголовка недостаточно видно
или форма вне перечисленных ниже, верни unknown, сохрани видимый заголовок.

Различай четыре формы по собственному заголовку и видимому контексту:
upd — универсальный передаточный документ; torg12 — товарная накладная формы
ТОРГ-12; ttn — товарно-транспортная накладная формы № 1-Т, включая её оборотную
сторону и продолжение товарного раздела (лист «Товарная накладная» с отметкой
«Страница 2» и/или строкой «Товарная накладная имеет продолжение на ___ листах,
на бланках за N ___» без кода ТОРГ-12); trn — отдельная транспортная накладная.
Строка ТОРГ-12 иная: «Товарная накладная имеет приложение на ___ листах и
содержит ___ порядковых номеров записей». Надпись «II. ТРАНСПОРТНЫЙ
РАЗДЕЛ» обозначает раздел документа и сама по себе не подтверждает отдельную
транспортную накладную. Читай заголовок самого листа, обозначение формы и
окружающий контекст, а не ссылку на другой документ, соседнюю бумагу или
заголовок одного раздела. Не восстанавливай невидимую форму по содержанию
таблицы, вероятности или по другим кадрам.

Запроси участки заголовка и контекста всех кадров вместе через zoom_regions:
один вызов со списком {frame_id, left, top, right, bottom}. Все участки будут
возвращены в нативном разрешении. Для одного дополнительного участка можно
использовать zoom_source с теми же полями.
frame_id — предоставленная метка frame_01 и т.д.; координаты [0,1] относятся
к ЦЕЛОМУ ПОВЁРНУТОМУ изображению именно этого кадра. Выбирай реальные нативные
участки собственного заголовка и контекста. panel_ids должны ссылаться только
на изображения, реально полученные для данного кадра. При отказе tools новое
изображение не выдано; неизвестное остаётся unknown.
У тебя не более 12 запросов и 12 изображений увеличения на совместный проход.
Последний запрос оставлен для submit_conflict_review. Заверши через этот tool,
вернув все отдельные чтения и resolution_reason. Не выбирай большинство и
не заставляй нечитаемые или разные кадры согласиться ради общего ответа.
"""


class _FrameCropRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: str
    left: Coordinate
    top: Coordinate
    right: Coordinate
    bottom: Coordinate


class _FrameField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: str = Field(description="Supplied frame_01/frame_02 label, never another frame's identity")
    status: Literal["recognized", "not_present", "unreadable", "conflict"]
    raw_value: str | None = Field(max_length=2000, description="Exact value freshly read on this frame; null for unresolved or absent")
    explicit_blank: bool = Field(description="True only for an explicitly visible blank own field")
    complete_page_visible: bool = Field(description="Whether the complete own page is visible in THIS source frame")
    evidence_text: str = Field(max_length=4000, description="Verbatim role/caption and value from THIS frame, without repairs")
    panel_ids: list[str] = Field(max_length=12, description="Actual supplied globally unique panel IDs belonging to THIS frame")
    visibility_cause: Literal["readable", "explicit_blank", "other_document_side", "occluded",
        "blur_or_small_text", "needs_more_zoom", "heading_uncertain", "unknown"] = Field(default="unknown",
        description="Fresh pixel assessment for THIS field on THIS frame. other_document_side only when the complete readable own side/continuation structurally lacks this field and obtaining another document side is necessary; never infer it from blur, clipping, occlusion, insufficient zoom, or uncertainty")


class _FieldReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resolution_reason: str = Field(max_length=2000, description="Explain the fresh pixel reading or disagreement; do not vote")
    readings: list[_FrameField] = Field(min_length=2, max_length=MAX_REVIEW_FRAMES,
        description="Exactly one independent fresh reading for EACH supplied frame")
    next_evidence_action: Literal["read_more_views", "obtain_other_document_side", "none"] = Field(default="read_more_views",
        description="obtain_other_document_side only when fresh supplied pixels show this complete side/continuation cannot contain the requested field; keep unreadable and request another document side. Ordinary unreadability, blur, occlusion, incomplete page, or conflicting readings require read_more_views")


class _FrameTypeReading(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: str = Field(description="Supplied frame_01/frame_02 label, never another frame's identity")
    document_type: DocumentType = Field(description="Own form freshly read on THIS frame; unknown when its own heading/context is insufficient or outside the four supported forms")
    heading_text: str = Field(max_length=4000, description="Literal own document heading/form identification visible on THIS frame; never substitute a section heading or another frame's text")
    panel_ids: list[str] = Field(max_length=12, description="Actual supplied globally unique panel IDs belonging to THIS frame")


class _TypeReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resolution_reason: str = Field(max_length=2000, description="Explain the fresh own-heading reading on all frames or the unresolved difference; never vote")
    readings: list[_FrameTypeReading] = Field(min_length=2, max_length=MAX_REVIEW_FRAMES,
        description="Exactly one fresh own-form reading for EACH supplied frame")


def _response_model(document_type: str, requested: list[str]):
    if document_type == "unknown":
        if requested:
            raise ValueError("Type review requires no requested fields")
        return _TypeReviewResponse
    fields_model = create_model("RequestedConflictFields", __config__=ConfigDict(extra="forbid"),
        **{name: (_FieldReview, Field(description=_FIELD_MODELS[document_type].model_fields[name].description))
           for name in requested})
    return create_model("ConflictReviewResponse", __config__=ConfigDict(extra="forbid"),
        document_type=(DocumentType, Field(description="Own form confirmed by at least two readable supplied frames; unreadable frames stay unreadable, but any actual different supported form means unknown")),
        fields=(fields_model, ...))


def _review_type(decision, sources: dict[str, _Source]):
    """Require two grounded agreeing source headings; never vote away a conflict."""
    issues, evidence, used = [], [], []
    labels = [reading.frame_id for reading in decision.readings]
    if len(labels) != len(sources) or set(labels) != set(sources):
        issues.append("Type review must contain each supplied frame exactly once")
    for reading in decision.readings:
        source = sources.get(reading.frame_id)
        if source is None:
            issues.append("Type reading names a frame that was not supplied")
            continue
        panel_ids = reading.panel_ids
        valid = bool(panel_ids) and all(identifier in source.proof["panels"]
            and source.proof["panels"][identifier]["frame_id"] == source.proof["frame_id"]
            and source.proof["panels"][identifier]["source_sha256"] == source.digest
            and source.proof["panels"][identifier]["panel_id"] == identifier for identifier in panel_ids)
        if panel_ids and not valid:
            issues.append("Type reading cites an unsupplied panel or another frame's panel")
        if valid:
            evidence.extend(Evidence(frame_id=source.proof["frame_id"], text=reading.heading_text,
                bbox=source.proof["panels"][identifier]["bbox"])
                for identifier in dict.fromkeys(panel_ids))
            if reading.document_type != "unknown" and reading.heading_text.strip():
                used.append(source)
        if reading.document_type != "unknown" and (not reading.heading_text.strip() or not valid):
            issues.append("Supported type reading requires a nonempty own heading and its own supplied panels")
    independence = {"frame_ids": sorted({source.proof["frame_id"] for source in used}),
        "source_sha256": sorted({source.digest for source in used}),
        "timestamp_s": sorted({source.proof["timestamp_s"] for source in used})}
    identities = [(source.proof["frame_id"], source.digest, source.proof["timestamp_s"]) for source in used]
    independent = any(all(a != b for a, b in zip(first, second))
        for index, first in enumerate(identities) for second in identities[index + 1:])
    if not independent:
        issues.append("Type corroboration requires distinct actual frame IDs, source SHAs and captured timestamps")
    types = {reading.document_type for reading in decision.readings if reading.document_type != "unknown"}
    agreed = next(iter(types)) if len(types) == 1 else "unknown"
    resolved = not issues and agreed != "unknown"
    kind = agreed if resolved else "unknown"
    return kind, evidence, {"status": "resolved" if resolved else "unresolved",
        "resolved_document_type": kind if resolved else None, "errors": list(dict.fromkeys(issues)),
        "independence": independence, "model_review": decision.model_dump(mode="json"),
        "method": "Fresh joint own-heading pixel review; two distinct readable source views must agree; unknown views do not confirm or veto, any supported contradictory form remains unresolved; no majority or previous type answers"}


def _review_fields(decision, sources: dict[str, _Source], document_type: str, requested: list[str]):
    """Validate actual citations and agreement; never reinterpret source text."""
    fields, details, errors = {}, {}, []
    for name in requested:
        review = getattr(decision.fields, name)
        issues, evidence, keys, absence, explicit_conflict = [], [], [], 0, False
        labels = [reading.frame_id for reading in review.readings]
        if len(labels) != len(sources) or set(labels) != set(sources):
            issues.append("Review must contain each supplied frame exactly once")
        used, recognized_sources, blank_sources = [], [], []
        for reading in review.readings:
            source = sources.get(reading.frame_id)
            if source is None:
                issues.append("Reading names a frame that was not supplied")
                continue
            panel_ids = reading.panel_ids
            valid = bool(panel_ids) and all(identifier in source.proof["panels"]
                and source.proof["panels"][identifier]["frame_id"] == source.proof["frame_id"]
                and source.proof["panels"][identifier]["source_sha256"] == source.digest
                and source.proof["panels"][identifier]["panel_id"] == identifier for identifier in panel_ids)
            if panel_ids and not valid:
                issues.append("Reading cites an unsupplied panel or another frame's panel")
            if valid:
                evidence.extend(Evidence(frame_id=source.proof["frame_id"], text=reading.evidence_text,
                    bbox=source.proof["panels"][identifier]["bbox"])
                    for identifier in dict.fromkeys(panel_ids))
                used.append(source)
            if reading.status == "recognized":
                if reading.raw_value is None or not reading.raw_value.strip() or reading.explicit_blank:
                    issues.append("Recognized reading requires nonempty raw_value and explicit_blank=false")
                elif not valid:
                    issues.append("Recognized reading requires its own supplied panels")
                else:
                    keys.append(_field_comparison(name, reading.raw_value))
                    recognized_sources.append(source)
            elif reading.status == "not_present":
                if (reading.raw_value is not None or not reading.explicit_blank
                        or not reading.complete_page_visible or not valid):
                    issues.append("Absent reading requires null raw_value, explicit blank, complete page and its own supplied panels")
                else:
                    absence += 1
                    blank_sources.append(source)
            else:
                if reading.raw_value is not None or reading.explicit_blank:
                    issues.append("Unresolved reading requires null raw_value and explicit_blank=false")
                explicit_conflict |= reading.status == "conflict"
        independence = {"frame_ids": sorted({source.proof["frame_id"] for source in used}),
            "source_sha256": sorted({source.digest for source in used}),
            "timestamp_s": sorted({source.proof["timestamp_s"] for source in used})}
        # Only readable supporting views can corroborate a value/absence. A
        # cited unreadable third image neither supplies confirmation nor vetoes
        # two genuinely distinct agreeing readable images.
        support = recognized_sources if keys else blank_sources
        identities = [(source.proof["frame_id"], source.digest, source.proof["timestamp_s"]) for source in support]
        independent = any(all(a != b for a, b in zip(first, second))
            for index, first in enumerate(identities) for second in identities[index + 1:])
        independence["supporting_frame_ids"] = sorted({source.proof["frame_id"] for source in support})
        if not independent and (len(keys) >= 2 or absence >= 2):
            issues.append("Corroboration requires distinct actual frame IDs, source SHAs and captured timestamps")
        different_type = decision.document_type != document_type
        if different_type:
            issues.append("Joint review did not confirm the requested form")
        if issues or different_type:
            status = "unreadable"
        elif explicit_conflict or len(set(keys)) > 1 or (keys and absence):
            status = "conflict"
        elif len(keys) >= 2 and independent:
            status = "recognized"
        elif absence >= 2 and independent:
            status = "not_present"
        else:
            status = "unreadable"
        raw = next((item.raw_value for item in review.readings if item.status == "recognized"), None) if status == "recognized" else None
        fields[name] = FieldValue(status=status, raw_value=raw,
            normalized_value=normalize(name, raw) if raw is not None else None,
            evidence=evidence, reason=review.resolution_reason)
        used_identities = [(source.proof["frame_id"], source.digest, source.proof["timestamp_s"]) for source in used]
        independent_unreadable = any(all(a != b for a, b in zip(first, second))
            for index, first in enumerate(used_identities) for second in used_identities[index + 1:])
        needs_other_side = (not issues and not different_type and status == "unreadable"
            and review.next_evidence_action == "obtain_other_document_side"
            and independent_unreadable and len(used) == len(review.readings)
            and all(item.status == "unreadable" and item.visibility_cause == "other_document_side"
                    and item.complete_page_visible and item.raw_value is None and not item.explicit_blank
                    and bool(item.evidence_text.strip())
                    for item in review.readings))
        details[name] = {"status": status, "resolved": status in {"recognized", "not_present"},
            "errors": list(dict.fromkeys(issues)), "independence": independence,
            "model_review": review.model_dump(mode="json"),
            "additional_evidence": {"action": review.next_evidence_action,
                "terminal_for_current_leaf": needs_other_side,
                "frame_ids": sorted({source.proof["frame_id"] for source in used}) if needs_other_side else [],
                "method": "Fresh joint source-pixel assessment, not independent model calls; missing values remain unreadable"}}
        errors.extend(f"{name}: {issue}" for issue in dict.fromkeys(issues))
    resolved = [name for name in requested if details[name]["resolved"]]
    return fields, {"status": "resolved" if len(resolved) == len(requested) else "unresolved",
        "resolved_fields": resolved, "unresolved_fields": [name for name in requested if name not in resolved],
        "field_reviews": details, "errors": errors,
        "method": "Fresh joint pixel review; two distinct readable source views must agree; unreadable views do not confirm or veto, every readable contradiction remains unresolved; no majority or previous answers"}


async def read_leaf_conflicts(frames: list[dict], *, source_id: str, page_id: str,
                             document_type: str, unresolved_fields: list[str],
                             config: AgentConfig, cache_dir: Path, trace_path: Path,
                             model: Model | None = None) -> dict:
    """Review 2–4 actual frames jointly, using only pixels and contested names.

    This is a fresh model-backed review, not independent model runs or measured
    accuracy. Only resolved fields may replace a caller's prior unresolved fields.
    document_type='unknown' and no requested fields instead reviews own forms,
    without supplying previous types, proposed answers, or a preferred type.
    """
    requested = list(unresolved_fields)
    type_review = document_type == "unknown" and not requested
    if not type_review and (document_type not in _FIELD_MODELS or not 1 <= len(requested) <= 6
            or len(set(requested)) != len(requested)
            or any(name not in FIELDS_BY_TYPE[document_type] for name in requested)):
        raise ValueError("Review requires unique requested fields of one supported form")
    if not 2 <= len(frames) <= MAX_REVIEW_FRAMES:
        raise ValueError("Joint review requires two to four actual source frames")
    for frame in frames:
        timestamp = frame.get("timestamp_s")
        if (not isinstance(frame.get("frame_id"), str) or not frame["frame_id"].strip()
                or not isinstance(frame.get("source_sha256"), str)
                or isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("Every frame requires actual ID, source SHA and captured timestamp")
    cache_dir, trace_path = Path(cache_dir).resolve(), Path(trace_path)
    sources = {f"frame_{index:02d}": _Source(frame, cache_dir, frame["rotation_clockwise_degrees"])
               for index, frame in enumerate(frames, 1)}
    effective = replace(config, model_name=config.vision_model_name, usage=config.usage, allow_tools=True)
    model_identity = getattr(model, "model_name", "") if model else effective.model_name
    response_model = _response_model(document_type, requested)
    instructions = TYPE_INSTRUCTIONS if type_review else INSTRUCTIONS
    prompt = ("Fresh own-form pixel review. Read the actual own document heading on each supplied frame."
              if type_review else "Fresh pixel review. Requested form=" + document_type + "; requested fields=" + _dump(requested))
    metadata = {"version": IMAGE_CONFLICTS_VERSION, "response_reuse": False, "semantic_validation": "model",
        "review_mode": "document_type" if type_review else "fields",
        "source_id": source_id, "page_id": page_id, "document_type": document_type,
        "requested_fields": requested, "model": model_identity, "base_url": effective.base_url,
        "provider_kind": effective.provider_kind, "model_request_limit": MAX_MODEL_REQUESTS,
        "zoom_image_limit": MAX_ZOOM_IMAGES, "output_limit": effective.max_output_tokens,
        "prompt_sha256": _sha((instructions + prompt).encode()),
        "response_schema_sha256": _sha(_dump(response_model.model_json_schema()).encode())}
    evidence_path = cache_dir / f"image-conflicts-{uuid4().hex}.json"
    deps = _PageDeps({}, source_id, page_id, trace_path, effective)
    run_usage = RunUsage()
    raw_responses, tool_proofs, tool_declines = [], [], []
    owned_client, completed = None, False

    def delta():
        return {key: getattr(run_usage, key) for key in ("requests", "input_tokens", "output_tokens", "tool_calls")}

    def proof():
        return {"sources": {label: source.proof for label, source in sources.items()},
            "panels": {identifier: panel for source in sources.values()
                       for identifier, panel in source.proof["panels"].items()}}

    def checkpoint():
        _write_evidence(evidence_path, {"status": "in_progress", "metadata": metadata, "proof": proof(),
            "raw_responses": raw_responses, "tool_proofs": tool_proofs, "tool_declines": tool_declines,
            "usage_delta": delta(), "origin": EVIDENCE_ORIGIN})

    def unchanged():
        for source in sources.values():
            source.check_unchanged()

    def register(label, identifier, panel):
        source = sources[label]
        global_id = label + "_" + identifier
        panel.update(panel_id=global_id, model_frame_id=label)
        del source.proof["panels"][identifier]
        source.proof["panels"][global_id] = panel
        return global_id

    async def before_request(ctx, request_context):
        unchanged()
        if deps.requests_started >= MAX_MODEL_REQUESTS:
            raise AnalysisError("Joint-review per-review request limit exhausted; continue other views")
        return await _before_request(ctx, request_context)

    async def after_request(ctx, request_context, response):
        raw_responses.append({"parts": [asdict(part) for part in response.parts], "usage": asdict(response.usage)})
        response = await _after_request(ctx, request_context, response)
        checkpoint()
        return response

    try:
        inputs = [prompt]
        for label, source in sources.items():
            identifier, data, panel = source.panel((0, 0, *source.upright_size), context=True)
            identifier = register(label, identifier, panel)
            inputs.extend([f"Actual frame_id={label}; panel_id={identifier}; whole upright source; pixels={panel['upload_pixel_size']}",
                           BinaryContent(data, media_type="image/jpeg")])
        _trace(trace_path, "image_conflicts_observation", source_id, page_id,
               metadata=metadata, evidence_path=str(evidence_path), proof=proof(), origin=EVIDENCE_ORIGIN)
        if (effective.usage.input_tokens >= effective.input_tokens_limit
                or effective.usage.output_tokens >= effective.output_tokens_limit):
            raise BatchBudgetExceeded("Shared joint-review token budget exhausted; no call made")
        if model is None:
            if not effective.model_name or not effective.base_url or (effective.provider_kind != "local" and not effective.api_key):
                raise AnalysisError("Explicit model and provider endpoint/key are required; no model call made")
            owned_client = AsyncOpenAI(api_key=effective.api_key or "local-no-key", base_url=effective.base_url,
                                       max_retries=0, timeout=effective.timeout_seconds)
            model = _RouterAIChatModel(effective.model_name, provider=OpenAIProvider(openai_client=owned_client))
        agent = Agent(model, output_type=ToolOutput(response_model, name="submit_conflict_review", strict=False),
            deps_type=_PageDeps, instructions=instructions, retries=0,
            name="sever_joint_type_review" if type_review else "sever_joint_field_review",
            capabilities=[Hooks(before_model_request=before_request, after_model_request=after_request)])

        async def prepare_zoom(ctx, tool_definition):
            started = max(effective.usage.requests, effective.usage.attempted_requests)
            if (len(tool_proofs) >= MAX_ZOOM_IMAGES or deps.requests_started >= MAX_MODEL_REQUESTS - 1
                    or (effective.max_requests or effective.request_limit) - started <= 1):
                return None
            return tool_definition

        @agent.tool_plain(sequential=True, retries=1, prepare=prepare_zoom)
        def zoom_source(frame_id: str, left: Coordinate, top: Coordinate,
                        right: Coordinate, bottom: Coordinate) -> ToolReturn:
            """Supply a native crop of the named whole upright source frame."""
            unchanged()
            source = sources.get(frame_id)
            if source is None:
                raise ModelRetry("Choose a supplied frame_01/frame_02 label.")
            if left >= right or top >= bottom:
                raise ModelRetry("Use left < right and top < bottom within the whole upright frame.")
            if len(tool_proofs) >= MAX_ZOOM_IMAGES:
                decline = {"frame_id": frame_id, "status": "zoom_limit_reached", "no_image_returned": True,
                           "requested_upright_bbox_xyxy": [left, top, right, bottom]}
                tool_declines.append(decline)
                _trace(trace_path, "image_conflicts_zoom_declined", source_id, page_id,
                       evidence_path=str(evidence_path), **decline)
                checkpoint()
                return ToolReturn(return_value=decline, content="No new image supplied. Submit or preserve unreadable.")
            width, height = source.upright_size
            box = (math.floor(left * width), math.floor(top * height), math.ceil(right * width), math.ceil(bottom * height))
            identifier, data, panel = source.panel(box, context=False, requested=[left, top, right, bottom])
            identifier = register(frame_id, identifier, panel)
            tool_proofs.append(panel)
            _trace(trace_path, "image_conflicts_zoom", source_id, page_id,
                   evidence_path=str(evidence_path), proof=panel, origin=EVIDENCE_ORIGIN)
            checkpoint()
            return ToolReturn(return_value={"frame_id": frame_id, "panel_id": identifier,
                "upright_bbox_xyxy": [left, top, right, bottom]},
                content=[f"Actual frame_id={frame_id}; panel_id={identifier}; native upright crop; pixels={panel['upload_pixel_size']}",
                         BinaryContent(data, media_type="image/png")], metadata=panel)

        @agent.tool_plain(sequential=True, retries=1, prepare=prepare_zoom)
        def zoom_regions(regions: Annotated[list[_FrameCropRegion], Field(min_length=1, max_length=MAX_ZOOM_IMAGES)]) -> ToolReturn:
            """Show native regions from several actual source frames together.

            Args:
                regions: Each rectangle names its supplied frame_01/frame_02 and normalized left, top, right, bottom in that whole upright frame.
            """
            # Validate the entire request before any panel is written or any
            # allowance consumed; a later malformed region must be atomic.
            for region in regions:
                if region.frame_id not in sources:
                    raise ModelRetry("Every region must name a supplied frame_01/frame_02 label.")
                if region.left >= region.right or region.top >= region.bottom:
                    raise ModelRetry("Every region needs left < right and top < bottom in its whole upright frame.")
            panels, contents, proofs = [], [], []
            for region in regions:
                supplied = zoom_source(**region.model_dump())
                panels.append(supplied.return_value)
                if isinstance(supplied.content, list):
                    contents.extend(supplied.content)
                elif supplied.content:
                    contents.append(supplied.content)
                if supplied.metadata:
                    proofs.append(supplied.metadata)
            return ToolReturn(return_value={"panels": panels}, content=contents,
                metadata={"source_panels": proofs})

        run = await agent.run(inputs, deps=deps, usage=run_usage,
            usage_limits=UsageLimits(request_limit=MAX_MODEL_REQUESTS),
            model_settings={"max_tokens": effective.max_output_tokens, "temperature": 0,
                            "tool_choice": "auto", "timeout": effective.timeout_seconds})
        unchanged()
        decision = run.output
        if type_review:
            agreed_type, classification_evidence, adjudication = _review_type(decision, sources)
            fields = {}
        else:
            fields, adjudication = _review_fields(decision, sources, document_type, requested)
        raw_decision = decision.model_dump(mode="json", exclude_unset=True)
        recognition = {"engine": f"native_image_conflicts:{model_identity}", "raw_decision": raw_decision,
            "raw_responses": raw_responses, "usage_delta": delta(),
            "response_reuse": False, "semantic_validation": "model", "pixel_proof": proof(),
            "model_requests": delta()["requests"], "zoom_images": len(tool_proofs),
            "errors": adjudication["errors"], "evidence_path": str(evidence_path), "metadata": metadata}
        result = {"fields": fields, "recognition": recognition, "panels": proof()["panels"], "adjudication": adjudication}
        type_result = ({"document_type": agreed_type,
                        "classification_evidence": [item.model_dump(mode="json") for item in classification_evidence]}
                       if type_review else {})
        if type_review:
            result.update(type_result)
        _trace(trace_path, "image_conflicts_reading", source_id, page_id, evidence_path=str(evidence_path),
               decision=raw_decision, fields={name: field.model_dump(mode="json") for name, field in fields.items()},
               adjudication=adjudication, usage_delta=delta(), origin=EVIDENCE_ORIGIN, **type_result)
        _write_evidence(evidence_path, {"status": "success", "metadata": metadata, "proof": proof(),
            "raw_decision": raw_decision, "fields": {name: field.model_dump(mode="json") for name, field in fields.items()},
            "adjudication": adjudication, "raw_responses": raw_responses, "tool_proofs": tool_proofs,
            "tool_declines": tool_declines, "usage_delta": delta(), "origin": EVIDENCE_ORIGIN, **type_result})
        completed = True
        return result
    except Exception as exc:
        _settle_run_usage(effective, run_usage, deps)
        status = getattr(exc, "status_code", None)
        fatal = isinstance(exc, FatalProviderError) or status in {400, 401, 402, 403}
        usage = effective.usage
        exhausted = (isinstance(exc, BatchBudgetExceeded)
            or (isinstance(exc, UsageLimitExceeded) and
                max(usage.requests, usage.attempted_requests) >= (effective.max_requests or effective.request_limit))
            or usage.input_tokens >= effective.input_tokens_limit or usage.output_tokens >= effective.output_tokens_limit
            or (effective.max_tokens is not None and usage.input_tokens + usage.output_tokens >= effective.max_tokens))
        proof_error = None
        try:
            _trace(trace_path, "image_conflicts_error", source_id, page_id, error_type=type(exc).__name__,
                   evidence_path=str(evidence_path), http_status=status, usage_delta=delta())
        except Exception as storage_error:
            proof_error = storage_error
        if not exhausted or deps.requests_started or raw_responses:
            try:
                _write_evidence(evidence_path, {"status": "budget_exhausted" if exhausted else "fatal" if fatal else "failed",
                    "metadata": metadata, "proof": proof(), "raw_responses": raw_responses,
                    "tool_proofs": tool_proofs, "tool_declines": tool_declines, "error_type": type(exc).__name__,
                    "http_status": status, "usage_delta": delta(), "origin": EVIDENCE_ORIGIN})
            except Exception as storage_error:
                proof_error = proof_error or storage_error
        if exhausted:
            raise BatchBudgetExceeded("Shared joint-review budget exhausted; no automatic retry") from None
        if fatal:
            raise FatalProviderError("Joint-review provider failure; stop batch") from None
        if proof_error is not None:
            raise EvidencePersistenceError(f"Cannot preserve joint-review proof ({type(proof_error).__name__})") from None
        raise AnalysisError(f"Joint pixel review failed ({type(exc).__name__}); no automatic retry") from None
    finally:
        _close_run_usage(effective, run_usage)
        if owned_client is not None:
            try:
                await owned_client.close()
            except Exception as close_error:
                if completed:
                    raise AnalysisError(f"Cannot close joint-review client ({type(close_error).__name__}); stop processing") from None
