"""Blind still-image runner: real input identities, no OCR/labels or network."""
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from sever.agents import AgentConfig, BatchBudgetExceeded
from sever.schema import DocumentResult, Evidence, FIELDS_BY_TYPE, FieldValue

spec = importlib.util.spec_from_file_location("blind_reference", Path(__file__).parents[1] / "tools/read_reference_images.py")
blind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blind)


def setup(tmp_path, count=2):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for number in range(count):
        Image.new("RGB", (50 + number, 80), (number * 80, 100, 200)).save(inputs / f"private-form-hint-{number}.png")
    return SimpleNamespace(input_dir=inputs, out=tmp_path / "out", model="offline-local",
                           max_requests=100, max_tokens=1_000_000)


def config(**changes):
    return AgentConfig(model_name="offline-local", vision_model_name="offline-local", provider_kind="local",
                       base_url="http://127.0.0.1:8090/v1", document_reader="native_image", **changes)


def answer(frame, source_id, page_id, kind="unknown", *, errors=None):
    label = "CMR" if kind == "unknown" else "Транспортная накладная"
    proof = {"frame_id": frame["frame_id"], "source_path": frame["path"], "source_sha256": source_id}
    fields = {name: FieldValue(status="unreadable") for name in FIELDS_BY_TYPE[kind]}
    return {"document": DocumentResult(source_id=source_id, page_id=page_id, document_type=kind,
        classification_evidence=[Evidence(frame_id=frame["frame_id"], text=label, bbox=[0, 0, 1, 1])],
        fields=fields, errors=errors or []),
        "recognition": {"observed_form": label, "visible_text": label + "\nLiteral source text",
                        "pixel_proof": proof}, "panels": {"context_r0": {"bbox": [0, 0, 1, 1]}}}


def test_blind_reads_exact_image_hashes_and_saves_unknown_literal_text(tmp_path, monkeypatch):
    args = setup(tmp_path)
    # The runner must never read an adjacent labels JSON or enter the old gates.
    (args.input_dir / "ground-truth.json").write_text("not valid json; forbidden label material")
    def forbidden(*a, **kw):
        raise AssertionError("OCR, separate classifier and fusion are forbidden")
    monkeypatch.setattr("sever.ocr.read_image", forbidden)
    monkeypatch.setattr("sever.classifier.classify_pages", forbidden)
    monkeypatch.setattr("sever.image_fusion.fuse_readings", forbidden)
    calls = []
    original = config()
    async def reader(frame, *, source_id, page_id, config, **kw):
        assert source_id == hashlib.sha256(Path(frame["path"]).read_bytes()).hexdigest()
        assert config.refresh_cache and config.provider_kind == "local"
        assert config.model_name == config.vision_model_name == args.model
        assert config.usage is original.usage
        assert "timestamp_s" not in frame
        assert frame["frame_id"] == "image-sha256-" + source_id
        calls.append(deepcopy(frame))
        return answer(frame, source_id, page_id)
    result = asyncio.run(blind.run(args, config=original, reader=reader))
    assert len(calls) == 2
    assert result["status"] == "completed"
    assert result["summary"]["types"] == {"unknown": 2}
    assert result["summary"]["accuracy_evaluated"] is False
    assert result["acceptance"]["independent_video_corroboration"] is False
    assert all(item["result"]["document_type"] == "unknown" for item in result["images"])
    assert all(item["recognition"]["visible_text"] == "CMR\nLiteral source text" for item in result["images"])
    assert json.loads((args.out / "results.json").read_text()) == result
    assert json.loads((args.out / "partial-results.json").read_text()) == result
    assert json.loads((args.out / "input-provenance.json").read_text()) == result["sources"]
    assert len({item["frame_id"] for item in result["images"]}) == 2


@pytest.mark.parametrize("mismatch", ["source", "page", "evidence_frame", "proof_sha", "proof_frame", "proof_path"])
def test_foreign_reading_is_rejected_and_failed_checkpoint_is_kept(tmp_path, mismatch):
    args = setup(tmp_path, count=1)
    async def reader(frame, *, source_id, page_id, **kw):
        result = answer(frame, source_id, page_id)
        if mismatch == "source":
            result["document"].source_id = "foreign-source"
        elif mismatch == "page":
            result["document"].page_id = "foreign-page"
        elif mismatch == "evidence_frame":
            result["document"].classification_evidence[0].frame_id = "foreign-frame"
        elif mismatch == "proof_sha":
            result["recognition"]["pixel_proof"]["source_sha256"] = "foreign-sha"
        elif mismatch == "proof_frame":
            result["recognition"]["pixel_proof"]["frame_id"] = "foreign-frame"
        else:
            result["recognition"]["pixel_proof"]["source_path"] = str(tmp_path / "foreign.png")
        return result
    with pytest.raises(ValueError):
        asyncio.run(blind.run(args, config=config(), reader=reader))
    saved = json.loads((args.out / "partial-results.json").read_text())
    assert saved["status"] == "failed" and saved["images"] == []
    assert saved["failure"]["error_type"] == "ValueError"
    assert not (args.out / "results.json").exists()


def test_validations_errors_remain_single_image_proposals_without_consensus(tmp_path):
    args = setup(tmp_path, count=1)
    async def reader(frame, *, source_id, page_id, **kw):
        result = answer(frame, source_id, page_id, "trn", errors=["Unreadable own header"])
        result["document"].fields["consignor"] = FieldValue(status="recognized", raw_value='ООО «Пример»',
            normalized_value='ООО «Пример»', evidence=[Evidence(frame_id=frame["frame_id"], text='Грузоотправитель: ООО «Пример»', bbox=[0, 0, 1, 1])])
        return result
    result = asyncio.run(blind.run(args, config=config(), reader=reader))
    assert result["status"] == "completed_with_errors"
    assert result["images"][0]["result"]["errors"] == ["Unreadable own header"]
    assert result["summary"]["recognized_single_image_proposals"] == 1
    assert result["images"][0]["single_image_proposal"] is True
    assert result["images"][0]["independent_video_corroboration"] is False


def test_provider_budget_stop_preserves_completed_images_without_publishing_final_result(tmp_path):
    args = setup(tmp_path)
    calls = []
    async def reader(frame, *, source_id, page_id, **kw):
        calls.append(frame["frame_id"])
        if len(calls) == 2:
            raise BatchBudgetExceeded("Shared cap reached")
        return answer(frame, source_id, page_id)
    with pytest.raises(BatchBudgetExceeded):
        asyncio.run(blind.run(args, config=config(), reader=reader))
    saved = json.loads((args.out / "partial-results.json").read_text())
    assert saved["status"] == "failed" and saved["summary"]["images_read"] == 1
    assert saved["failure"]["error_type"] == "BatchBudgetExceeded"
    assert not (args.out / "results.json").exists()


def test_source_mutation_after_reading_cannot_be_published(tmp_path):
    args = setup(tmp_path, count=1)
    async def reader(frame, *, source_id, page_id, **kw):
        result = answer(frame, source_id, page_id)
        Image.new("RGB", (40, 50), "red").save(frame["path"])
        return result
    with pytest.raises(ValueError, match="changed while"):
        asyncio.run(blind.run(args, config=config(), reader=reader))
    assert not (args.out / "results.json").exists()


def test_local_only_configuration_rejects_external_provider_and_missing_explicit_alias(tmp_path, monkeypatch):
    args = setup(tmp_path, count=1)
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "remote")
    with pytest.raises(ValueError, match="external fallback is disabled"):
        blind.local_config(args)
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    monkeypatch.setenv("HARNESS_MODEL_BASE_URL", "https://routerai.ru/api/v1")
    with pytest.raises(ValueError, match="loopback"):
        blind.local_config(args)
    monkeypatch.delenv("HARNESS_MODEL_NAME", raising=False)
    with pytest.raises(SystemExit):
        blind.arguments(["--input-dir", str(args.input_dir), "--out", str(args.out)])


def test_config_uses_explicit_cli_model_and_forces_fresh_native_reading(tmp_path, monkeypatch):
    args = setup(tmp_path, count=1)
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "local")
    monkeypatch.setenv("HARNESS_MODEL_BASE_URL", "http://127.0.0.1:8090/v1")
    monkeypatch.setenv("HARNESS_VISION_MODEL", "a-different-env-model")
    cfg = blind.local_config(args)
    assert cfg.model_name == cfg.vision_model_name == args.model
    assert cfg.refresh_cache and cfg.document_reader == "native_image"
    assert not cfg.allow_tools and not cfg.verify_inn


def test_existing_output_is_not_reused_for_a_claimed_fresh_reading(tmp_path):
    args = setup(tmp_path, count=1)
    args.out.mkdir()
    (args.out / "trace.jsonl").write_text("previous proof")
    async def forbidden(*a, **kw):
        raise AssertionError("No reader call is allowed")
    with pytest.raises(ValueError, match="new or empty"):
        asyncio.run(blind.run(args, config=config(), reader=forbidden))
    assert (args.out / "trace.jsonl").read_text() == "previous proof"


@pytest.mark.parametrize("failed_artifact", ["summary.json", "results.json", "completed_trace"])
def test_failed_finalization_withdraws_success_artifacts_and_records_completed_readings(tmp_path, monkeypatch, failed_artifact):
    args = setup(tmp_path, count=1)
    real_save, real_trace = blind.save_json, blind.trace
    def save(path, value):
        if Path(path).name == failed_artifact:
            Path(path).write_text('{"status":"completed"}')
            raise OSError("Controlled output failure")
        real_save(path, value)
    def trace(path, event, **data):
        if failed_artifact == "completed_trace" and event == "blind_reference_run_completed":
            raise OSError("Controlled trace failure")
        real_trace(path, event, **data)
    monkeypatch.setattr(blind, "save_json", save)
    monkeypatch.setattr(blind, "trace", trace)
    async def reader(frame, *, source_id, page_id, config, **kw):
        config.usage.requests += 1
        config.usage.input_tokens += 100
        return answer(frame, source_id, page_id)
    with pytest.raises(OSError, match="Controlled"):
        asyncio.run(blind.run(args, config=config(), reader=reader))
    partial = json.loads((args.out / "partial-results.json").read_text())
    assert partial["status"] == "failed"
    assert partial["failure"]["stage"] == "finalization"
    assert partial["summary"]["images_read"] == 1
    assert partial["usage"]["requests"] == 1 and partial["usage"]["input_tokens"] == 100
    assert not (args.out / "results.json").exists()
    assert not (args.out / "summary.json").exists()
    rows = [json.loads(line) for line in (args.out / "trace.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "blind_reference_run_failed"
    assert not any(row["event"] == "blind_reference_run_completed" for row in rows)


def test_blind_runner_defaults_to_same_zoom_reader_as_video(tmp_path, monkeypatch):
    args = setup(tmp_path, count=1)
    monkeypatch.setenv("SEVER_RUNS_DIR", str(tmp_path / "runs"))
    seen = []
    async def reader(frame, *, source_id, page_id, **kwargs):
        seen.append(frame["frame_id"])
        return answer(frame, source_id, page_id)
    async def forbidden(*args, **kwargs):
        raise AssertionError("A different still-image recipe was selected")
    monkeypatch.setattr("sever.zoom_image_reader.read_frame_zoom", reader)
    monkeypatch.setattr("sever.adaptive_image_reader.read_frame_adaptive", forbidden)
    result = asyncio.run(blind.run(args, config=config()))
    assert result["status"] == "completed" and len(seen) == 1


def test_blind_runner_cannot_read_inputs_or_call_model_during_private_evaluation(tmp_path, monkeypatch):
    from sever.inference_gate import evaluation_access, gate_path, InferenceGateBusy
    args = setup(tmp_path, count=1)
    monkeypatch.setenv("SEVER_RUNS_DIR", str(tmp_path / "runs"))
    def forbidden(*args, **kwargs):
        raise AssertionError("Inputs must not be read during evaluation")
    monkeypatch.setattr(blind, "image_sources", forbidden)
    with evaluation_access(gate_path()):
        with pytest.raises(InferenceGateBusy):
            asyncio.run(blind.run(args, config=config()))
    assert not args.out.exists()
