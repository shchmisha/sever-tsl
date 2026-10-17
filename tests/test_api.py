from concurrent.futures import ThreadPoolExecutor
import json
import platform
from pathlib import Path
import shutil
import subprocess
import time

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from pydantic_ai import ModelResponse
from pydantic_ai.messages import TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage
import pytest

from sever import api, classifier, vision
from sever.agents import AgentConfig
from sever.api import app

def test_health_and_missing_source():
    with TestClient(app) as client:
        assert client.get("/health").status_code==200
        assert client.post("/v1/jobs",json={"source_path":"/missing.mp4"}).status_code==400
        assert client.get("/v1/jobs/../../etc/passwd").status_code==404

def test_unconfigured_job_fails_before_queued(monkeypatch,tmp_path):
    monkeypatch.setenv("HARNESS_PROVIDER_KIND","remote")
    monkeypatch.delenv("HARNESS_MODEL_API_KEY",raising=False)
    video=tmp_path/"video.mp4";video.write_bytes(b"not a video")
    assert TestClient(app).post("/v1/jobs",json={"source_path":str(video)}).status_code==503


def test_local_video_job_completes_with_offline_model(monkeypatch, tmp_path):
    """Real API worker/video/local OCR/artifacts; FunctionModel is NOT live LLM coverage."""
    if not shutil.which("ffmpeg"):
        pytest.skip("Local video integration requires FFmpeg")
    native = platform.system() == "Darwin" and bool(shutil.which("swiftc"))
    if not native:
        tesseract = shutil.which("tesseract")
        if not tesseract:
            pytest.skip("Local integration requires Apple Vision or Tesseract with rus+eng")
        languages = subprocess.run([tesseract, "--list-langs"], capture_output=True, text=True, check=True).stdout
        if not {"rus", "eng"}.issubset({line.strip() for line in languages.splitlines()}):
            pytest.skip("Local integration requires Tesseract rus+eng language data")
    image = tmp_path / "certificate.png"
    paper = Image.new("RGB", (640, 960), "white")
    draw = ImageDraw.Draw(paper)
    draw.text((30, 40), "QUALITY CERTIFICATE / OFFLINE API FIXTURE", fill="black", font_size=24)
    for row in range(16):
        y = 120 + row * 40
        draw.line((30, y, 610, y), fill="black", width=2)
        draw.text((40, y + 5), f"LOT {row:03d} PRODUCT {100000 + row * 139} QUALITY PASSED", fill="black", font_size=18)
    paper.save(image)
    source = tmp_path / "certificate.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-loop", "1", "-i", str(image),
                    "-t", "1.6", "-r", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True)
    calls = []

    def respond(messages, info):
        prompt = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)][-1]
        data = json.loads(prompt.split("\n", 1)[1])
        calls.append(data)
        decisions = [{"page_id": page["page_id"], "document_type": "unknown",
                      "classification_evidence_ids": [page["regions"][0]["id"]] if page["regions"] else [],
                      "observation": "Controlled offline response for the certificate fixture"}
                     for page in data["pages"]]
        return ModelResponse(parts=[TextPart(json.dumps({"pages": decisions}))],
                             usage=RequestUsage(input_tokens=100, output_tokens=50))

    offline_model = FunctionModel(respond, model_name="offline-api-fixture")
    real_classify = classifier.classify_pages

    async def offline_classify(pages, **kwargs):
        return await real_classify(pages, model=offline_model, **kwargs)

    async def forbid_reader(*args, **kwargs):
        raise AssertionError("An unknown offline fixture must not call a vision provider")

    configs = []

    def offline_config(model, max_requests, max_tokens):
        config = AgentConfig(model_name="offline-api-fixture", max_requests=max_requests, max_tokens=max_tokens)
        configs.append(config)
        return config

    monkeypatch.setattr(api, "ROOT", tmp_path / "runs")
    monkeypatch.setattr(api, "agent_config", offline_config)
    monkeypatch.setattr(classifier, "classify_pages", offline_classify)
    monkeypatch.setattr(vision, "read_document", forbid_reader)
    with ThreadPoolExecutor(max_workers=1) as worker:
        monkeypatch.setattr(api, "WORKER", worker)
        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200 and health.json()["status"] == "ok"
            queued = client.post("/v1/jobs", json={"source_path": str(source), "sample_interval": .5,
                                                 "max_pages": 5, "max_requests": 5, "max_tokens": 100000})
            assert queued.status_code == 202
            job_id = queued.json()["job_id"]
            deadline = time.monotonic() + 60
            while True:
                status = client.get(f"/v1/jobs/{job_id}").json()
                if status["status"] in {"completed", "completed_with_errors", "failed"}:
                    break
                assert time.monotonic() < deadline, status
                time.sleep(.025)
            assert status["status"] == "completed", status
            results = client.get(f"/v1/jobs/{job_id}/results")
            assert results.status_code == 200
            payload = results.json()
            assert payload["pages"] and all(page["result"]["document_type"] == "unknown" for page in payload["pages"])
            assert payload["source"]["path"] == str(source.resolve())
            csv = client.get(f"/v1/jobs/{job_id}/fields.csv")
            assert csv.status_code == 200 and "unknown" in csv.text
    assert len(calls) == configs[0].usage.requests == 1
    job_dir = api.ROOT / job_id
    assert (job_dir / "report.html").is_file()
    assert (job_dir / "classifications.json").is_file()
    observations = [view for path in (job_dir / "ocr").glob("*.json") for view in json.loads(path.read_text())]
    assert observations and all(view["regions"] for view in observations)
    expected_engine = "apple_vision" if native else "tesseract_rus_eng"
    assert all(view["engine"].startswith(expected_engine) for view in observations)
    assert all(Path(frame["path"]).is_file() for page in payload["pages"] for frame in page["frames"])
    assert any(json.loads(line)["event"] == "run_completed" for line in (job_dir / "trace.jsonl").read_text().splitlines())
