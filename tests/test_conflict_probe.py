import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from sever.agents import AgentConfig, BatchBudgetExceeded, FatalProviderError
from sever.inference_gate import InferenceGateBusy, evaluation_access
from sever.schema import Evidence, FieldValue

spec = importlib.util.spec_from_file_location("conflict_probe", Path(__file__).parents[1] / "tools/read_conflict_images.py")
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("SEVER_INFERENCE_GATE_PATH", str(tmp_path / "gate.lock"))
    frames = []
    for index in range(2):
        path = tmp_path / f"actual-frame-{index}.png"
        Image.new("RGB", (40, 60), (index * 50, 100, 200)).save(path)
        frames.append({"frame_id": f"actual-{index}", "path": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "timestamp_s": index + .5,
            "rotation_clockwise_degrees": index * 90, "source_pts": index + 1, "source_time_base": "1/1000"})
    manifest = {"source_id": "video-source-sha", "page_id": "leaf-id", "document_type": "upd",
                "fields": ["consignor_address"], "frames": frames}
    input_path = tmp_path / "manifest.json"
    input_path.write_text(json.dumps(manifest))
    return SimpleNamespace(input=input_path, out=tmp_path / "new-output", model="offline-local",
        max_requests=24, max_tokens=1_000_000), manifest


def config(**changes):
    return AgentConfig(model_name="offline-local", vision_model_name="offline-local", provider_kind="local",
        base_url="http://127.0.0.1:8090/v1", document_reader="native_image", **changes)


def answer(frames, requested):
    fields = {name: FieldValue(status="recognized", raw_value="  Буквальный адрес  ", normalized_value=None,
        evidence=[Evidence(frame_id=frame["frame_id"], text="", bbox=[0, 0, 1, 1]) for frame in frames]) for name in requested}
    return {"fields": fields, "adjudication": {"status": "resolved", "resolved_fields": requested},
        "recognition": {"raw_decision": {"untouched": "fresh model reply"}, "raw_responses": [{"fresh": True}],
            "pixel_proof": {"sources": {f"frame_{index:02d}": {"frame_id": frame["frame_id"],
                "source_path": frame["path"], "source_sha256": frame["source_sha256"],
                "timestamp_s": frame["timestamp_s"], "rotation_clockwise_degrees": frame["rotation_clockwise_degrees"]}
                for index, frame in enumerate(frames, 1)}}}, "panels": {}}


def test_manifest_only_fresh_reader_shared_gate_and_lossless_serialization(setup):
    args, manifest = setup
    seen, original = [], config()
    (args.input.parent / "ground-truth.json").write_text("DO-NOT-READ")
    async def reader(frames, *, config, unresolved_fields, **kw):
        assert config.usage is original.usage and config.refresh_cache and config.provider_kind == "local"
        assert config.model_name == config.vision_model_name == args.model
        assert config.max_requests == 24 and config.max_tokens == 1_000_000
        with pytest.raises(InferenceGateBusy):
            with evaluation_access(): pass
        seen.append(deepcopy(frames))
        config.usage.requests += 1
        config.usage.input_tokens += 50
        config.usage.output_tokens += 20
        return answer(frames, unresolved_fields)
    result = asyncio.run(probe.run(args, config=original, reader=reader))
    assert seen == [manifest["frames"]]
    assert result["status"] == "completed" and result["elapsed_seconds"] >= 0
    assert result["answer"]["fields"]["consignor_address"]["raw_value"] == "  Буквальный адрес  "
    assert result["answer"]["recognition"]["raw_responses"] == [{"fresh": True}]
    assert result["usage"]["requests"] == 1 and result["usage"]["input_tokens"] == 50
    assert result["acceptance"]["accuracy_evaluated"] is False
    assert result["acceptance"]["independent_model_corroboration"] is False
    assert json.loads((args.out / "results.json").read_text()) == result
    assert json.loads((args.out / "partial-results.json").read_text()) == result
    with evaluation_access(): pass


def test_type_only_probe_passes_no_suggested_form_or_fields(setup):
    args, manifest = setup
    manifest.update(document_type="unknown", fields=[])
    args.input.write_text(json.dumps(manifest))

    async def reader(frames, *, document_type, unresolved_fields, **kwargs):
        assert document_type == "unknown" and unresolved_fields == []
        output = answer(frames, [])
        output.update(document_type="ttn", classification_evidence=[Evidence(
            frame_id=frame["frame_id"], text="Own printed form heading", bbox=[0, 0, 1, 1]
        ).model_dump(mode="json") for frame in frames])
        return output

    result = asyncio.run(probe.run(args, config=config(), reader=reader))
    assert result["answer"]["document_type"] == "ttn"
    assert result["answer"]["fields"] == {}
    assert len(result["answer"]["classification_evidence"]) == 2


@pytest.mark.parametrize("where", ["top", "frame", "wrong_field", "wrong_rotation", "wrong_type", "bad_pts"])
def test_manifest_excludes_prior_answers_and_non_source_hints(setup, where):
    args, manifest = setup
    if where == "top": manifest["expected_values"] = {"consignor_address": "Injected answer"}
    elif where == "frame": manifest["frames"][0]["previous_reading"] = {"raw_value": "Injected answer"}
    elif where == "wrong_field": manifest["fields"] = ["expected_value"]
    elif where == "wrong_rotation": manifest["frames"][0]["rotation_clockwise_degrees"] = 45
    elif where == "bad_pts": manifest["frames"][0]["timestamp_s"] = True
    else: manifest["document_type"] = "unknown"
    args.input.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        asyncio.run(probe.run(args, config=config(), reader=lambda *a, **kw: pytest.fail("Model called")))
    assert not args.out.exists()


def test_exclusive_evaluation_gate_refuses_before_any_read(setup):
    args, _ = setup
    with evaluation_access():
        with pytest.raises(InferenceGateBusy):
            asyncio.run(probe.run(args, config=config(), reader=lambda *a, **kw: pytest.fail("Model called")))
    assert not args.out.exists()


def test_nonempty_output_refuses_replay_and_does_not_touch_existing_file(setup):
    args, _ = setup
    args.out.mkdir()
    historical = args.out / "results.json"
    historical.write_text("HISTORICAL-ANSWER")
    with pytest.raises(ValueError, match="empty output"):
        asyncio.run(probe.run(args, config=config(), reader=lambda *a, **kw: pytest.fail("Model called")))
    assert historical.read_text() == "HISTORICAL-ANSWER"


def test_hash_mismatch_refuses_before_reader(setup):
    args, manifest = setup
    manifest["frames"][0]["source_sha256"] = "0" * 64
    args.input.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA"):
        asyncio.run(probe.run(args, config=config(), reader=lambda *a, **kw: pytest.fail("Model called")))
    assert not args.out.exists()


@pytest.mark.parametrize("mismatch", ["field_contract", "evidence", "proof", "source_mutation"])
def test_foreign_answer_or_mutated_source_keeps_failed_checkpoint(setup, mismatch):
    args, _ = setup
    async def reader(frames, *, unresolved_fields, **kw):
        result = answer(frames, unresolved_fields)
        if mismatch == "field_contract": result["fields"]["extra"] = FieldValue(status="unreadable")
        elif mismatch == "evidence": result["fields"]["consignor_address"].evidence[0].frame_id = "foreign-frame"
        elif mismatch == "proof": result["recognition"]["pixel_proof"]["sources"]["frame_01"]["source_sha256"] = "foreign"
        else: Image.new("RGB", (40, 60), "red").save(frames[0]["path"])
        return result
    with pytest.raises(ValueError):
        asyncio.run(probe.run(args, config=config(), reader=reader))
    saved = json.loads((args.out / "partial-results.json").read_text())
    assert saved["status"] == "failed" and saved["failure"]["error_type"] == "ValueError"
    assert not (args.out / "results.json").exists()


@pytest.mark.parametrize("error", [BatchBudgetExceeded, FatalProviderError, RuntimeError])
def test_failure_status_real_usage_and_proof_survive_without_hiding_exception(setup, error):
    args, _ = setup
    original = config()
    async def reader(frames, *, cache_dir, config, **kw):
        cache_dir.mkdir()
        (cache_dir / "actual-received-proof.json").write_text('{"received":true}')
        config.usage.requests += 1
        config.usage.input_tokens += 33
        raise error("Sensitive provider body must not enter runner failure")
    with pytest.raises(error):
        asyncio.run(probe.run(args, config=original, reader=reader))
    saved = json.loads((args.out / "partial-results.json").read_text())
    assert saved["status"] == ("budget_exhausted" if error is BatchBudgetExceeded else "failed")
    assert saved["usage"]["requests"] == 1 and saved["usage"]["input_tokens"] == 33
    assert (args.out / "image-reading-evidence" / "actual-received-proof.json").is_file()
    assert "Sensitive provider body" not in json.dumps(saved)
    assert not (args.out / "results.json").exists()
    with evaluation_access(): pass


@pytest.mark.parametrize("error", [BatchBudgetExceeded, FatalProviderError])
def test_failure_checkpoint_storage_does_not_mask_primary_fatal_or_budget(setup, monkeypatch, error):
    args, _ = setup
    save = probe.save_json
    def storage(path, payload):
        if payload.get("status") in {"failed", "budget_exhausted"}: raise OSError("disk full")
        return save(path, payload)
    monkeypatch.setattr(probe, "save_json", storage)
    async def reader(*a, **kw): raise error("original")
    with pytest.raises(error, match="original"):
        asyncio.run(probe.run(args, config=config(), reader=reader))


def test_finalization_failure_never_leaves_completed_result(setup, monkeypatch):
    args, _ = setup
    save = probe.save_json
    def storage(path, payload):
        if path.name == "results.json": raise OSError("no space")
        return save(path, payload)
    monkeypatch.setattr(probe, "save_json", storage)
    async def reader(frames, *, unresolved_fields, **kw): return answer(frames, unresolved_fields)
    with pytest.raises(OSError):
        asyncio.run(probe.run(args, config=config(), reader=reader))
    saved = json.loads((args.out / "partial-results.json").read_text())
    assert saved["status"] == "failed" and saved["failure"]["stage"] == "finalization"
    assert "answer" in saved and not (args.out / "results.json").exists()


def test_local_config_and_cli_defaults_do_not_allow_remote_fallback(setup, monkeypatch):
    args, _ = setup
    monkeypatch.setenv("HARNESS_PROVIDER_KIND", "remote")
    with pytest.raises(ValueError, match="local"):
        asyncio.run(probe.run(args))
    remote = AgentConfig()
    with pytest.raises(ValueError, match="local"):
        asyncio.run(probe.run(args, config=remote))
    parsed = probe.arguments(["--input", str(args.input), "--out", str(args.out), "--model", "offline-local"])
    assert parsed.max_requests == 24 and parsed.max_tokens == 1_000_000
