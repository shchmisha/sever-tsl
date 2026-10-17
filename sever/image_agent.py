"""Fresh single-frame reading with model-selected native source crops.

The evidence directory is write-only. All images in a run come from one pinned
source snapshot; coordinates supplied by the model refer to its upright view.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from io import BytesIO
import math
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, BinaryContent, ModelRetry, ToolOutput, ToolReturn
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits
from openai import AsyncOpenAI

from .agents import (AgentConfig, AnalysisError, BatchBudgetExceeded, FatalProviderError,
                     _PageDeps, _RouterAIChatModel, _after_request, _before_request,
                     _dump, _trace, _write_cache as _write_evidence)
from .image_detail import _rotate, _source_box
from .image_reader import (EVIDENCE_ORIGIN, IMAGE_SCHEMA_VERSION, INSTRUCTIONS,
                           _NativeReadingResponse, _inverse_affine, _sha,
                           _source_transform, _validate_reading)

IMAGE_AGENT_VERSION = "fresh-model-selected-batched-source-crops-v6-compact"
MAX_MODEL_REQUESTS = 8
MAX_ZOOM_IMAGES = 8
Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class _SourceChanged(AnalysisError):
    pass


class _CropLimitExceeded(AnalysisError):
    pass


class ViewBudgetExceeded(AnalysisError):
    """This frame used its allowance; other frames can still be processed."""


class EvidencePersistenceError(AnalysisError):
    """Model evidence could not be saved; later readable views cannot repair it."""


class CropRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: Coordinate
    top: Coordinate
    right: Coordinate
    bottom: Coordinate

ZOOM_INSTRUCTIONS = """
You initially receive ONE upright context image of this frame, reduced to a
1024-pixel long edge. Use zoom_regions to request the heading and all needed
caption-and-value regions together in ONE call whenever their positions are
visible. This saves round trips while returning every region at native resolution.
Use zoom_source for a single additional region after examining those details.
Before the final reading, zoom the document's own heading/form code and the
relevant caption-and-value lines. Choose tight readable rectangles, preserving
the caption with its value. If a crop is insufficient, choose another rectangle
from the whole upright frame. Read abbreviations character by character from
the detail pixels; do not expand or repair them from a likely organization name.
If several papers overlap in the frame, use the own main paper's heading and
caption context; never combine another paper's values or roles into its reading.
The tool accepts normalized left, top, right, bottom in [0, 1], measured from the
top-left of the WHOLE UPRIGHT frame, never from a previous crop. Returned images
are lossless native crops, without magnification or invented pixels. Cite only
the supplied panel IDs that actually contain the quoted text. All panels belong
to this same single source frame and provide no independent corroboration.
You have at most 8 model requests and 8 crop images. Use the remaining budget
to return the structured reading; unresolved characters/roles stay unreadable.
When no more crops are available, submit using the pixels already received.
Check the complete reading against the supplied images before submitting:
For a supported form, leave visible_text empty and quote only each field's own
caption in its evidence_text; write the value once, in raw_value. Do not call a field
blank merely because its line was outside a crop. A footer naming
the document's creator does not by itself show consignor or consignee roles.
Submit the final structured reading through the submit_reading tool.
"""


class _Source:
    def __init__(self, frame: dict, folder: Path, rotation: int):
        if type(rotation) is not int or rotation not in (0, 90, 180, 270):
            raise ValueError("Rotation must be an integer 0, 90, 180 or 270")
        try:
            self.path = Path(frame["path"]).resolve()
            source_bytes = self.path.read_bytes()
            with Image.open(BytesIO(source_bytes)) as original:
                original.load()
                orientation = original.getexif().get(274)
                if original.mode in {"RGBA", "LA"} or "transparency" in original.info:
                    rgba = original.convert("RGBA")
                    self.image = Image.new("RGB", rgba.size, "white")
                    self.image.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    self.image = original.convert("RGB")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AnalysisError(f"Cannot read saved source frame ({type(exc).__name__}); no model call made") from None
        self.digest = _sha(source_bytes)
        if frame.get("source_sha256") is not None and frame["source_sha256"] != self.digest:
            raise AnalysisError("Saved source frame SHA does not match supplied provenance; no model call made")
        self.rotation, self.folder = rotation, folder / "image-panels"
        self.run_id, self.zoom_count = uuid4().hex, 0
        self.upright_size = self.image.size[::-1] if rotation % 180 else self.image.size
        self.proof = {"frame_id": str(frame.get("frame_id") or frame.get("id") or f"source-sha256-{self.digest}"),
            "source_path": str(self.path), "source_sha256": self.digest,
            "source_pixel_size": list(self.image.size), "source_exif_orientation": orientation,
            "upright_pixel_size": list(self.upright_size),
            "coordinate_system": "normalized_top_left_saved_source_frame",
            "tool_coordinate_system": "normalized_top_left_whole_upright_frame_xyxy",
            "rotation_clockwise_degrees": rotation, "recipe": IMAGE_AGENT_VERSION,
            "timestamp_s": frame.get("timestamp_s"),
            "display_rotation_applied": frame.get("display_rotation_applied"), "panels": {}}
        self.check_unchanged()

    def check_unchanged(self):
        try:
            unchanged = _sha(self.path.read_bytes()) == self.digest
        except OSError:
            unchanged = False
        if not unchanged:
            raise _SourceChanged("Source pixels changed during image-agent reading")

    def panel(self, upright_box: tuple[int, int, int, int], *, context: bool,
              requested: list[float] | None = None):
        self.check_unchanged()
        box = _source_box(upright_box, self.image.size, self.rotation)
        uploaded = _rotate(self.image.crop(box), self.rotation)
        if context:
            uploaded.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        if context:
            uploaded.save(buffer, "JPEG", quality=95, subsampling=0)
        else:
            uploaded.save(buffer, "PNG")
        data = buffer.getvalue()
        digest = _sha(data)
        identifier = f"context_r{self.rotation}" if context else f"zoom_{self.zoom_count + 1:02d}_r{self.rotation}"
        extension = "jpg" if context else "png"
        self.folder.mkdir(parents=True, exist_ok=True)
        target = self.folder / f"{self.run_id}-{identifier}-{digest}.{extension}"
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(data)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        width, height = self.image.size
        left, top, right, bottom = box
        transform = _source_transform(box, self.rotation, uploaded.size)
        panel = {"panel_id": identifier, "frame_id": self.proof["frame_id"],
            "source_sha256": self.digest,
            "bbox": [left / width, top / height, (right-left) / width, (bottom-top) / height],
            "source_bbox_pixels_top_left": list(box), "upright_bbox_pixels_top_left": list(upright_box),
            "requested_upright_bbox_xyxy": requested,
            "rotation_clockwise_degrees": self.rotation, "upload_pixel_size": list(uploaded.size),
            "upload_sha256": digest, "upload_path": str(target),
            "source_to_upload_pixels": transform, "upload_to_source_pixels": _inverse_affine(transform),
            "prep": {"format": "JPEG" if context else "PNG", "max_edge": 1024 if context else None,
                **({"quality": 95, "subsampling": 0} if context else {}),
                "resize": "LANCZOS" if context else "none", "native_source_detail": not context,
                "exif_orientation_applied": False, "recipe": IMAGE_AGENT_VERSION}}
        self.check_unchanged()
        self.proof["panels"][identifier] = panel
        if not context:
            self.zoom_count += 1
        return identifier, data, panel

    def zoom(self, left: float, top: float, right: float, bottom: float):
        coordinates = [left, top, right, bottom]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in coordinates) or left >= right or top >= bottom:
            raise ModelRetry("Use 0 <= left < right <= 1 and 0 <= top < bottom <= 1 in the whole upright frame.")
        if self.zoom_count >= MAX_ZOOM_IMAGES:
            raise _CropLimitExceeded("Image-agent crop limit exhausted; no further image supplied")
        width, height = self.upright_size
        box = (math.floor(left * width), math.floor(top * height),
               math.ceil(right * width), math.ceil(bottom * height))
        return self.panel(box, context=False, requested=coordinates)


async def read_frame_agent(frame: dict, *, source_id: str, page_id: str, config: AgentConfig,
                           cache_dir: Path, trace_path: Path, rotation: int,
                           model: Model | None = None) -> dict:
    """Read fresh pixels with a zoom tool; never load a historical answer."""
    cache_dir, trace_path = Path(cache_dir).resolve(), Path(trace_path)
    source = _Source(frame, cache_dir, rotation)
    effective = replace(config, model_name=config.vision_model_name, usage=config.usage, allow_tools=True)
    model_identity = getattr(model, "model_name", "") if model else effective.model_name
    instructions = INSTRUCTIONS + ZOOM_INSTRUCTIONS
    prompt = ("Read this actual upright frame. Select the heading and needed native regions "
              "together with zoom_regions; use zoom_source for a single follow-up crop if needed. "
              "Then return one form-and-fields reading, including unknown visible text.")
    metadata = {"version": IMAGE_AGENT_VERSION, "response_schema_version": IMAGE_SCHEMA_VERSION,
        "response_schema_sha256": _sha(_dump(_NativeReadingResponse.model_json_schema()).encode()),
        "model": model_identity, "base_url": effective.base_url, "provider_kind": effective.provider_kind,
        "source_id": source_id, "page_id": page_id, "source_sha256": source.digest,
        "frame_id": source.proof["frame_id"], "response_reuse": False, "semantic_validation": "model",
        "output_limit": effective.max_output_tokens, "model_request_limit": MAX_MODEL_REQUESTS,
        "zoom_image_limit": MAX_ZOOM_IMAGES, "prompt_sha256": _sha((instructions + prompt).encode())}
    evidence_path = cache_dir / f"image-agent-{source.run_id}.json"
    deps = _PageDeps({}, source_id, page_id, trace_path, effective)
    before = asdict(effective.usage)
    raw_responses, tool_proofs, tool_declines = [], [], []
    owned_client, completed = None, False

    def delta():
        return {key: asdict(effective.usage)[key] - before[key]
                for key in ("requests", "input_tokens", "output_tokens", "tool_calls")}

    def checkpoint():
        # Updating this run's output evidence never turns it into model input.
        _write_evidence(evidence_path, {"status": "in_progress", "metadata": metadata,
            "proof": source.proof, "raw_responses": raw_responses, "tool_proofs": tool_proofs, "tool_declines": tool_declines,
            "usage_delta": delta(), "origin": EVIDENCE_ORIGIN})

    async def before_request(ctx, request_context):
        source.check_unchanged()
        if deps.requests_started >= MAX_MODEL_REQUESTS:
            raise ViewBudgetExceeded("This frame's model request allowance exhausted")
        return await _before_request(ctx, request_context)

    async def after_request(ctx, request_context, response):
        raw_responses.append({"parts": [asdict(part) for part in response.parts],
                              "usage": asdict(response.usage)})
        response = await _after_request(ctx, request_context, response)
        checkpoint()
        return response

    try:
        identifier, data, panel = source.panel((0, 0, *source.upright_size), context=True)
        _trace(trace_path, "image_agent_observation", source_id, page_id, metadata=metadata,
               evidence_path=str(evidence_path), origin=EVIDENCE_ORIGIN, proof=source.proof)
        if (effective.usage.input_tokens >= effective.input_tokens_limit
                or effective.usage.output_tokens >= effective.output_tokens_limit):
            raise BatchBudgetExceeded("Shared image-reading token budget exhausted; no model call made")
        if model is None:
            if not effective.model_name or not effective.base_url or (effective.provider_kind != "local" and not effective.api_key):
                raise AnalysisError("Explicit model and provider endpoint/key are required; no image call made")
            owned_client = AsyncOpenAI(api_key=effective.api_key or "local-no-key", base_url=effective.base_url,
                                       max_retries=0, timeout=effective.timeout_seconds)
            model = _RouterAIChatModel(effective.model_name, provider=OpenAIProvider(openai_client=owned_client))
        agent = Agent(model, output_type=ToolOutput(_NativeReadingResponse, name="submit_reading", strict=False), deps_type=_PageDeps,
            instructions=instructions, retries=0, name="sever_source_zoom_agent",
            capabilities=[Hooks(before_model_request=before_request, after_model_request=after_request)])

        async def prepare_zoom(ctx, tool_definition):
            # Reserve the final request for submission. Never fabricate a crop
            # to satisfy a request after the real image allowance is exhausted.
            shared_started = max(effective.usage.requests, effective.usage.attempted_requests)
            shared_limit = effective.max_requests or effective.request_limit
            if (source.zoom_count >= MAX_ZOOM_IMAGES
                    or deps.requests_started >= MAX_MODEL_REQUESTS - 1
                    or shared_limit - shared_started <= 1):
                return None
            return tool_definition

        @agent.tool_plain(sequential=True, retries=1, prepare=prepare_zoom)
        def zoom_source(left: Coordinate, top: Coordinate, right: Coordinate, bottom: Coordinate) -> ToolReturn:
            """Show native pixels from a normalized rectangle of the whole upright frame.

            Args:
                left: Left edge, normalized from 0 to 1.
                top: Top edge, normalized from 0 to 1.
                right: Right edge, greater than left, up to 1.
                bottom: Bottom edge, greater than top, up to 1.
            """
            source.check_unchanged()
            if source.zoom_count >= MAX_ZOOM_IMAGES:
                # A single model response can contain more calls than the
                # remaining allowance. Return an explicit non-image result;
                # the next model request exposes only the submission tool.
                declined = {"status": "zoom_limit_reached", "no_image_returned": True,
                            "requested_upright_bbox_xyxy": [left, top, right, bottom]}
                tool_declines.append(declined)
                _trace(trace_path, "image_agent_zoom_declined", source_id, page_id,
                       evidence_path=str(evidence_path), **declined)
                checkpoint()
                return ToolReturn(return_value=declined,
                    content="No additional image supplied: crop allowance exhausted. Submit using the supplied pixels; unresolved fields stay unreadable.")
            crop_id, crop_bytes, crop_proof = source.zoom(left, top, right, bottom)
            tool_proofs.append(crop_proof)
            _trace(trace_path, "image_agent_zoom", source_id, page_id, evidence_path=str(evidence_path),
                   origin=EVIDENCE_ORIGIN, proof=crop_proof)
            checkpoint()
            description = (f"Actual source panel_id={crop_id}; native upright crop "
                           f"left={left}, top={top}, right={right}, bottom={bottom}; "
                           f"pixels={crop_proof['upload_pixel_size']}. Coordinates refer to the whole upright frame.")
            return ToolReturn(return_value={"panel_id": crop_id, "upright_bbox_xyxy": [left, top, right, bottom]},
                content=[description, BinaryContent(crop_bytes, media_type="image/png")], metadata=crop_proof)

        @agent.tool_plain(sequential=True, retries=1, prepare=prepare_zoom)
        def zoom_regions(regions: Annotated[list[CropRegion], Field(min_length=1, max_length=8)]) -> ToolReturn:
            """Show native regions together, each in normalized whole-frame coordinates.

            Args:
                regions: Rectangles with left, top, right, bottom from 0 to 1 in the whole upright frame.
            """
            # Validate all boxes before saving any panels or consuming allowance.
            for region in regions:
                if region.left >= region.right or region.top >= region.bottom:
                    raise ModelRetry("Each region must have left < right and top < bottom.")
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

        run = await agent.run([prompt, f"Actual source panel_id={identifier}; whole upright frame; "
            f"pixels={panel['upload_pixel_size']}", BinaryContent(data, media_type="image/jpeg")],
            deps=deps, usage=effective.usage,
            usage_limits=UsageLimits(request_limit=min(effective.max_requests or effective.request_limit,
                                                       effective.usage.requests + MAX_MODEL_REQUESTS),
                input_tokens_limit=effective.input_tokens_limit, output_tokens_limit=effective.output_tokens_limit,
                total_tokens_limit=effective.max_tokens),
            model_settings={"max_tokens": effective.max_output_tokens, "temperature": 0,
                            "tool_choice": "auto", "timeout": effective.timeout_seconds})
        source.check_unchanged()
        decision = run.output.reading
        result = _validate_reading(decision, source.proof, source_id=source_id, page_id=page_id,
                                   model_identity=model_identity)
        result["recognition"]["engine"] = f"native_image_agent:{model_identity}"
        result["recognition"]["model_requests"] = delta()["requests"]
        result["recognition"]["zoom_images"] = source.zoom_count
        _trace(trace_path, "image_agent_reading", source_id, page_id, evidence_path=str(evidence_path),
               decision=decision.model_dump(mode="json"), result=result["document"].model_dump(mode="json"),
               usage_delta=delta(), origin=EVIDENCE_ORIGIN)
        _write_evidence(evidence_path, {"status": "success", "metadata": metadata, "proof": source.proof,
            "decision": decision.model_dump(mode="json"), "result": result["document"].model_dump(mode="json"),
            "raw_responses": raw_responses, "tool_proofs": tool_proofs, "tool_declines": tool_declines, "usage_delta": delta(), "origin": EVIDENCE_ORIGIN})
        completed = True
        return result
    except Exception as exc:
        committed = effective.usage.requests - before["requests"]
        effective.usage.requests += max(0, deps.requests_started - committed)
        status = getattr(exc, "status_code", None)
        fatal = isinstance(exc, FatalProviderError) or status in {400, 401, 402, 403}
        # Response usage is charged by the SDK before after-request hooks. A
        # checkpoint failure must not hide an already exceeded token budget.
        usage = effective.usage
        token_overrun = (usage.input_tokens > effective.input_tokens_limit
            or usage.output_tokens > effective.output_tokens_limit
            or (effective.max_tokens is not None
                and usage.input_tokens + usage.output_tokens > effective.max_tokens))
        shared_limit = effective.max_requests or effective.request_limit
        shared_exhausted = max(usage.requests, usage.attempted_requests) >= shared_limit
        local_exhausted = (isinstance(exc, ViewBudgetExceeded) or (
            isinstance(exc, UsageLimitExceeded) and deps.requests_started >= MAX_MODEL_REQUESTS)
            ) and not token_overrun and not shared_exhausted
        exhausted = (isinstance(exc, BatchBudgetExceeded) or token_overrun
                     or (isinstance(exc, (UsageLimitExceeded, ViewBudgetExceeded)) and not local_exhausted))
        proof_error = None
        try:
            _trace(trace_path, "image_agent_error", source_id, page_id, error_type=type(exc).__name__,
                   evidence_path=str(evidence_path), http_status=status, usage_delta=delta())
        except Exception as storage_error:
            proof_error = storage_error
        # A started run keeps received responses and crop proofs on budget stops.
        # A preflight stop before any call still creates no answer record.
        if not exhausted or deps.requests_started or raw_responses:
            try:
                _write_evidence(evidence_path, {"status": "budget_exhausted" if exhausted else "view_budget_exhausted" if local_exhausted else "fatal" if fatal else "failed", "metadata": metadata,
                    "proof": source.proof, "raw_responses": raw_responses, "tool_proofs": tool_proofs, "tool_declines": tool_declines,
                    "error_type": type(exc).__name__, "http_status": status, "usage_delta": delta()})
            except Exception as storage_error:
                proof_error = proof_error or storage_error
        if exhausted:
            raise BatchBudgetExceeded("Shared image-agent budget exhausted; no automatic retry") from None
        if fatal:
            message = (f"Image provider rejected request (HTTP {status}); stop batch"
                       if status is not None else "Image provider failure; stop batch")
            raise FatalProviderError(message) from None
        if proof_error is not None:
            raise EvidencePersistenceError(f"Cannot preserve image-agent failure proof ({type(proof_error).__name__})") from None
        if local_exhausted:
            raise ViewBudgetExceeded("This frame's model request allowance exhausted; remaining job budget is preserved") from None
        if isinstance(exc, (_SourceChanged, _CropLimitExceeded)):
            raise AnalysisError(str(exc)) from None
        raise AnalysisError(f"Image-agent reading failed ({type(exc).__name__}); no automatic retry") from None
    finally:
        if owned_client is not None:
            try:
                await owned_client.close()
            except Exception as close_error:
                if completed:
                    raise AnalysisError(f"Cannot close image provider client ({type(close_error).__name__}); stop processing") from None
