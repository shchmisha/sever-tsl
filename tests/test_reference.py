"""Reference-runner wiring with controlled local OCR/model responses, no paid calls."""
import asyncio
import json
from types import SimpleNamespace

from PIL import Image
from pydantic_ai import ModelResponse
from pydantic_ai.messages import TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from sever.schema import FIELDS_BY_TYPE
from sever.vision import VisionDecision, VisionField
from tools import run_reference


def test_four_reference_forms_share_one_classification_and_four_pixel_calls(monkeypatch, tmp_path):
    """Checks real classifier/vision budget/cache wiring, not document-reading accuracy."""
    kinds = ["upd", "torg12", "ttn", "trn"]
    headings = ["Универсальный передаточный документ", "Товарная накладная ТОРГ-12",
                "Товарно-транспортная накладная", "Транспортная накладная"]
    source = tmp_path / "reference.pdf"
    source.write_bytes(b"%PDF-1.4\nOffline runner fixture; images supplied separately")
    for n in range(1, 5):
        Image.new("RGB", (400, 600), "white").save(tmp_path / f"ocr-page-{n}.jpg")
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"customer_approved": False, "pages": [
        {"page_id": f"reference-{n}", "start_s": n - 1, "end_s": n, "document_type": kinds[n - 1], "fields": {}}
        for n in range(1, 5)]}))

    def local_ocr(path, cache_dir, **kwargs):
        n = int(path.stem.split("-")[-1])
        return {"engine": "controlled-offline-ocr", "processing": {"ocr_image_path": str(path)},
                "regions": [{"text": headings[n - 1], "bbox": [.05, .05, .9, .05]}]}

    batch_calls, pixel_calls, configs = [], [], []

    def classify_response(messages, info):
        prompt = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)][-1]
        data = json.loads(prompt.split("\n", 1)[1])
        batch_calls.append(data)
        decisions = [{"page_id": page["page_id"], "document_type": kinds[int(page["page_id"].split("-")[-1]) - 1],
                      "classification_evidence_ids": [page["regions"][0]["id"]]}
                     for page in data["pages"]]
        return ModelResponse(parts=[TextPart(json.dumps({"pages": decisions}))],
                             usage=RequestUsage(input_tokens=150, output_tokens=70))

    real_classify, real_read = run_reference.classify_pages, run_reference.read_document

    async def classify(pages, **kwargs):
        configs.append(kwargs["config"])
        return await real_classify(pages, model=FunctionModel(classify_response, model_name="offline-reference-classifier"), **kwargs)

    async def pixels(pages, **kwargs):
        kind = kwargs["document_type"]
        configs.append(kwargs["config"])

        def response(messages, info):
            pixel_calls.append(kind)
            output = VisionDecision(document_type=kind, heading_text=headings[kinds.index(kind)],
                                    fields={name: VisionField(status="unreadable", reason="Controlled test response")
                                            for name in FIELDS_BY_TYPE[kind]})
            return ModelResponse(parts=[TextPart(output.model_dump_json())],
                                 usage=RequestUsage(input_tokens=100, output_tokens=60))

        return await real_read(pages, model=FunctionModel(response, model_name="offline-reference-vision"), **kwargs)

    monkeypatch.setenv("HARNESS_MODEL_API_KEY", "offline-key-marker")
    monkeypatch.setattr(run_reference, "read_image", local_ocr)
    monkeypatch.setattr(run_reference, "classify_pages", classify)
    monkeypatch.setattr(run_reference, "read_document", pixels)
    args = SimpleNamespace(source=source, out=tmp_path / "run", images_dir=tmp_path, truth=truth,
                           model="offline-reference-classifier", max_requests=5, max_tokens=100000)
    result = asyncio.run(run_reference.run(args))
    assert len(batch_calls) == 1 and len(batch_calls[0]["pages"]) == 4
    assert pixel_calls == kinds
    assert all(config is configs[0] for config in configs)
    assert result["usage"]["requests"] == 5
    assert [page["result"]["document_type"] for page in result["pages"]] == kinds
    assert all(set(page["result"]["fields"]) == set(FIELDS_BY_TYPE[kind]) for page, kind in zip(result["pages"], kinds))
    metrics = json.loads((args.out / "metrics.json").read_text())
    assert metrics["detection"]["rate"] is None
    assert metrics["selected_frame_quality"]["evaluated"] is False
    assert (args.out / "fields.csv").is_file() and (args.out / "report.html").is_file()
    assert all("offline-key-marker" not in path.read_text() for path in args.out.rglob("*.json"))
