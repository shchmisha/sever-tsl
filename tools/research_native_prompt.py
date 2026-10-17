"""Bounded fresh local-pixel prompt comparison; never loads model answers or GT."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider

from sever.agents import AgentConfig, _RouterAIChatModel
from sever import image_reader


RUSSIAN_PREFIX = """Задача — извлечь значения реквизитов из настоящих изображений,
а не переписать указатели вместо значений. Для организационных полей результатом
должно быть видимое наименование организации. «Он же», «тот же» и подобные слова
являются ссылкой, а не наименованием организации или адресом.
Если строка «Грузоотправитель» содержит «он же», сначала прочитай на этой же
странице строку «Продавец» и её наименование. Затем верни в raw_value поля
consignor именно прочитанное наименование продавца, не слова «он же».
В evidence_text сохрани обе реально видимые строки: ссылку грузоотправителя и
продавца с наименованием; panel_ids должны содержать настоящие панели этих строк.
Для consignor_address при такой же явной ссылке прочитай именно видимый адрес
продавца, не адрес покупателя. Адреса переписывай как видны, без исправления
почтовых индексов, добавления региона, улицы или других отсутствующих частей.
Если связанное наименование или адрес не читается, пометь соответствующее поле
unreadable с raw_value=null. Ничего не восстанавливай по памяти и не додумывай.
Перед окончательным ответом проверь, что организационные raw_value содержат
прочитанные организации, а адресные raw_value — прочитанные адреса; объяснение
ссылки находится в evidence_text/reason, а не заменяет значение реквизита.
Все решения принимай только по предоставленным изображениям.

"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


class RequestOnlyModel(_RouterAIChatModel):
    def __init__(self, *args, thinking: bool | None, request_path: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.thinking = thinking
        self.request_path = request_path

    async def request(self, messages, model_settings, model_request_parameters):
        settings = dict(model_settings or {})
        if self.thinking is not None:
            settings["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.thinking}}
        save(self.request_path, {"model": self.model_name, "settings": settings,
            "message_count": len(messages), "saved_answers_input": False,
            "ground_truth_input": False, "schema_sha256": hashlib.sha256(
                json.dumps(model_request_parameters.output_object.json_schema,
                           ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
        return await super().request(messages, settings, model_request_parameters)


async def run(args):
    if args.out.exists():
        raise FileExistsError("Use a new isolated output directory")
    args.out.mkdir(parents=True)
    selected_bytes = args.selected_pages.read_bytes()
    pages = json.loads(selected_bytes)
    page = pages[0]
    matches = [frame for frame in page["frames"] if frame["frame_id"] == page["selected_frame_id"]]
    if len(matches) != 1:
        raise ValueError("Selected source frame must be present exactly once")
    frame = matches[0]
    source = Path(frame["path"]).resolve()
    source_sha = digest(source)
    original_instructions = image_reader.INSTRUCTIONS
    config = AgentConfig(model_name=args.model, vision_model_name=args.model,
        base_url=args.base_url, provider_kind="local", document_reader="native_image",
        timeout_seconds=1800, max_output_tokens=8192, request_limit=3, max_requests=3,
        input_tokens_limit=950_000, output_tokens_limit=50_000, max_tokens=1_000_000,
        refresh_cache=True, allow_tools=False, verify_inn=False)
    results = {"run_kind": "isolated_same_actual_frame_prompt_experiment", "status": "processing",
        "source": {"path": str(source), "sha256": source_sha, "frame_id": frame["frame_id"],
                   "timestamp_s": frame.get("timestamp_s"), "page_id": page["page_id"],
                   "selected_pages_sha256": hashlib.sha256(selected_bytes).hexdigest()},
        "configuration": {"model": args.model, "endpoint": args.base_url, "request_limit": 3,
                          "aggregate_token_limit": 1_000_000, "output_max_tokens": 8192},
        "ground_truth_read": False, "saved_answers_input": False,
        "independent_frame_corroboration": False, "accuracy_evaluated": False,
        "production_reader_changed": False, "experiments": []}
    save(args.out / "summary.json", results)
    client = AsyncOpenAI(api_key="local-no-key", base_url=args.base_url, max_retries=0, timeout=1800)
    try:
        variants = [("current", "", None), ("russian-entity-reference", RUSSIAN_PREFIX, None),
                    ("russian-entity-reference-thinking", RUSSIAN_PREFIX, True)]
        for name, prefix, thinking in variants:
            image_reader.INSTRUCTIONS = prefix + original_instructions
            folder = args.out / name
            folder.mkdir()
            (folder / "instructions.txt").write_text(image_reader.INSTRUCTIONS, encoding="utf-8")
            before = asdict(config.usage)
            started = time.monotonic()
            entry = {"variant": name, "thinking_override": thinking,
                     "instructions_sha256": hashlib.sha256(image_reader.INSTRUCTIONS.encode()).hexdigest(),
                     "status": "processing"}
            results["experiments"].append(entry)
            save(args.out / "summary.json", results)
            model = RequestOnlyModel(args.model, provider=OpenAIProvider(openai_client=client),
                                     thinking=thinking, request_path=folder / "request-settings.json")
            print(json.dumps({"started": name, "source_frame": frame["frame_id"]}), flush=True)
            try:
                answer = await image_reader.read_frame(frame, source_id=source_sha, page_id=page["page_id"],
                    config=config, cache_dir=folder / "evidence", trace_path=folder / "trace.jsonl", model=model)
                payload = {**answer, "document": answer["document"].model_dump(mode="json")}
                save(folder / "reading.json", payload)
                document = payload["document"]
                entry.update(status="completed", document_type=document["document_type"],
                             errors=document["errors"], fields=document["fields"])
            except Exception as error:
                entry.update(status="failed", error_type=type(error).__name__, error=str(error))
            entry["elapsed_seconds"] = round(time.monotonic() - started, 3)
            entry["usage_delta"] = {key: asdict(config.usage)[key] - before[key]
                                     for key in ("requests", "input_tokens", "output_tokens", "tool_calls")}
            if digest(source) != source_sha:
                raise ValueError("Actual input pixels changed during the experiment")
            results["usage"] = config.usage.model_dump()
            save(args.out / "summary.json", results)
            print(json.dumps({"finished": name, "status": entry["status"],
                "fields": {key: value["raw_value"] for key, value in entry.get("fields", {}).items()},
                "usage_delta": entry["usage_delta"]}, ensure_ascii=False), flush=True)
        results["status"] = "completed" if all(e["status"] == "completed" for e in results["experiments"]) else "completed_with_errors"
        save(args.out / "summary.json", results)
    finally:
        image_reader.INSTRUCTIONS = original_instructions
        await client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-pages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
