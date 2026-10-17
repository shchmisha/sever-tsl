"""Insert honest reading pauses into a completed native-image video's playback.

Only results.json, its source video and the actual model-read frame images supply
content. No OCR, expected-value table or manual field values are used. Boxes are
never drawn over document words: native evidence describes uploaded panels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

from sever.image_fusion import _field_comparison
from sever.schema import DocumentResult

LABELS = {
    "invoice_number": "Номер счёт-фактуры", "invoice_date": "Дата счёт-фактуры",
    "consignor": "Грузоотправитель", "consignor_address": "Адрес грузоотправителя",
    "consignee": "Грузополучатель", "consignee_address": "Адрес грузополучателя",
    "supplier": "Поставщик", "document_number": "Номер документа",
    "document_date": "Дата документа", "customer": "Заказчик перевозки",
}
VERSION = "native-reading-freezes-model-authority-v2"
SIZE = (1920, 1080)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_path(left, right) -> bool:
    return isinstance(left, str) and isinstance(right, str) and Path(left).resolve() == Path(right).resolve()


def _timestamp(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Source timestamp must be a finite nonnegative number")
    return float(value)


def _valid_box(value) -> bool:
    return (isinstance(value, list) and len(value) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)
        and value[0] >= 0 and value[1] >= 0 and value[2] > 0 and value[3] > 0
        and value[0] + value[2] <= 1.000001 and value[1] + value[3] <= 1.000001)


def _frame_evidence(items, frame_id, panels):
    for item in items:
        if item.frame_id != frame_id or item.region_index is not None:
            raise ValueError("Native evidence must belong to this actual read frame")
        if not _valid_box(item.bbox) or not any(
            panel.get("frame_id") == frame_id and panel.get("bbox") == item.bbox for panel in panels.values()
        ):
            raise ValueError("Native evidence box is not a recorded uploaded source panel")


def _read_view(page, view, source_id):
    frame_id = view.get("frame_id")
    matches = [frame for frame in page["frames"] if frame.get("frame_id") == frame_id]
    if not isinstance(frame_id, str) or not frame_id or len(matches) != 1:
        raise ValueError("Reading is not bound to one actual frame of the selected leaf")
    frame = matches[0]
    proof = view.get("pixel_proof", {})
    if proof.get("frame_id") != frame_id or not _same_path(frame.get("path"), view.get("frame_path")) or not _same_path(frame.get("path"), proof.get("source_path")):
        raise ValueError("Reading frame paths or source pixel proof disagree")
    path = Path(frame["path"]).resolve()
    if sha256(path) != proof.get("source_sha256"):
        raise ValueError("Actual model-read frame bytes fail their recorded SHA256")
    timestamp = _timestamp(view.get("timestamp_s"))
    if any(abs(timestamp - _timestamp(value)) > 1e-7 for value in (frame.get("timestamp_s"), proof.get("timestamp_s"))):
        raise ValueError("Reading timestamp differs from its recorded source frame")
    with Image.open(path) as image:
        if list(image.size) != proof.get("source_pixel_size"):
            raise ValueError("Actual model-read frame dimensions differ from pixel proof")
    document = DocumentResult.model_validate(view["document"])
    if document.source_id != source_id or document.page_id != page["page_id"]:
        raise ValueError("Reading document belongs to a different source or leaf")
    if view.get("origin") != "model_pixel_transcription":
        raise ValueError("Only native pixel readings can supply model provenance")
    eligible = not document.errors and document.document_type == page["result"]["document_type"]
    panels = proof.get("panels", {})
    if not isinstance(panels, dict) or not panels:
        raise ValueError("Reading lacks actual uploaded-panel provenance")
    def valid_panel(identifier):
        panel = panels.get(identifier)
        return (panel is not None and panel.get("panel_id") == identifier
                and panel.get("frame_id") == frame_id)
    _frame_evidence(document.classification_evidence, frame_id, panels)
    if eligible and not document.classification_evidence:
        raise ValueError("Reading has no own-heading evidence on this source frame")
    original = view.get("reading", {})
    if original.get("document_type") != document.document_type:
        raise ValueError("Native document type differs from its original model reading")
    for name, value in document.fields.items():
        _frame_evidence(value.evidence, frame_id, panels)
        if value.status == "recognized":
            if not value.raw_value.strip() or not value.evidence:
                raise ValueError("Recognized native value lacks raw value or actual source-panel evidence")
            proposal = original.get("fields", {}).get(name, {})
            if proposal.get("status") != "recognized" or proposal.get("raw_value") != value.raw_value:
                raise ValueError("Recognized native value differs from its original model reading")
            ids = proposal.get("panel_ids", [])
            if not ids or any(not valid_panel(identifier) for identifier in ids):
                raise ValueError("Recognized native value refers to an unavailable input panel")
            if any(not any(panels[identifier]["bbox"] == evidence.bbox for identifier in ids)
                   for evidence in value.evidence):
                raise ValueError("Recognized native evidence differs from the model's supplied panels")
    heading_ids = original.get("heading_panel_ids", [])
    if eligible and (not heading_ids or any(not valid_panel(identifier) for identifier in heading_ids)):
        raise ValueError("Reading heading refers to an unavailable input panel")
    located_heading_ids = [identifier for identifier in heading_ids if valid_panel(identifier)]
    if any(not any(panels[identifier]["bbox"] == evidence.bbox for identifier in located_heading_ids)
           for evidence in document.classification_evidence):
        raise ValueError("Heading evidence differs from the model's supplied panels")
    angles = {panels[identifier].get("rotation_clockwise_degrees") for identifier in located_heading_ids}
    if eligible and (not angles or any(angle not in {0, 90, 180, 270} for angle in angles)):
        raise ValueError("Reading heading does not provide a valid display orientation")
    return {"frame_id": frame_id, "path": str(path), "source_frame_sha256": proof["source_sha256"],
        "timestamp_s": timestamp, "source_pixel_size": proof["source_pixel_size"],
        "heading_panel_ids": heading_ids, "heading_rotations": sorted(angles),
        "confirmed_candidate": eligible,
        "document": document.model_dump(mode="json"), "panels": panels,
        "source_pts": frame.get("source_pts"), "source_time_base": frame.get("source_time_base"),
        "display_rotation_applied": frame.get("display_rotation_applied")}


def build_plan(run: Path, *, page_id: str | None = None, freeze_seconds: float = 1.5) -> dict:
    """Validate source/view proof and select two fully supporting actual reads."""
    if not isinstance(freeze_seconds, (int, float)) or not math.isfinite(freeze_seconds) or freeze_seconds <= 0:
        raise ValueError("Freeze duration must be finite and positive")
    run = Path(run).resolve()
    results_path = run / "results.json"
    result_bytes = results_path.read_bytes()
    data = json.loads(result_bytes)
    configuration = data.get("configuration", {})
    if configuration.get("document_reader") != "native_image" or configuration.get("local_ocr_used") is not False:
        raise ValueError("This renderer requires completed direct image readings")
    source = Path(data["source"]["path"]).resolve()
    source_id = data["source"]["sha256"]
    if sha256(source) != source_id:
        raise ValueError("Source video bytes fail results.json SHA256")
    pages = [page for page in data["pages"] if (page_id is None or page["page_id"] == page_id)
             and any(value.get("status") == "recognized" for value in page.get("result", {}).get("fields", {}).values())]
    if len(pages) != 1:
        raise ValueError("Select exactly one leaf containing recognized results using --page")
    page = pages[0]
    final = DocumentResult.model_validate(page["result"])
    if final.source_id != source_id or final.page_id != page["page_id"]:
        raise ValueError("Final recognized result belongs to a different source leaf")
    recognized = {name: value for name, value in final.fields.items() if value.status == "recognized"}
    views = [_read_view(page, view, source_id) for view in page.get("image_recognition", {}).get("views", [])]
    if len({view["frame_id"] for view in views}) != len(views):
        raise ValueError("Repeated frame IDs cannot count as independent readings")
    by_id = {view["frame_id"]: view for view in views}
    all_final_evidence = [*final.classification_evidence, *(e for field in final.fields.values() for e in field.evidence)]
    for evidence in all_final_evidence:
        view = by_id.get(evidence.frame_id)
        if view is None:
            raise ValueError("Final result contains evidence from a foreign/unread frame")
        _frame_evidence([evidence], view["frame_id"], view["panels"])
    final_type_ids = {e.frame_id for e in final.classification_evidence}
    if len({view["frame_id"] for view in views
            if view["confirmed_candidate"] and view["frame_id"] in final_type_ids}) < 2:
        raise ValueError("Final type lacks two distinct actual own-heading readings")
    support = {}
    for name, value in recognized.items():
        final_ids = {e.frame_id for e in value.evidence}
        supporting = []
        for view in views:
            if not view["confirmed_candidate"]:
                continue
            field = view["document"]["fields"][name]
            if (view["frame_id"] in final_ids and field["status"] == "recognized"
                    and _field_comparison(name, field["raw_value"]) == _field_comparison(name, value.raw_value)
                    and field["evidence"]):
                supporting.append(view["frame_id"])
        if len(set(supporting)) < 2:
            raise ValueError(f"{name} lacks two distinct actual frame readings of the recognized value")
        support[name] = supporting
    common_ids = set.intersection(*(set(ids) for ids in support.values()))
    candidates = sorted((view for view in views if view["frame_id"] in common_ids), key=lambda view: view["timestamp_s"])
    if len(candidates) < 2:
        raise ValueError("No two actual frame readings support every rendered recognized field")
    selected = candidates[:2]
    common_rotations = set.intersection(*(set(view["heading_rotations"]) for view in selected))
    if len(common_rotations) != 1:
        raise ValueError("Own-heading panels do not identify one common display rotation")
    rotation = common_rotations.pop()
    for view in selected:
        view["fields"] = {name: {"raw_value": view["document"]["fields"][name]["raw_value"],
                "fused_raw_value": value.raw_value, "evidence": view["document"]["fields"][name]["evidence"],
                "confirmed_frame_ids": support[name]} for name, value in recognized.items()}
    return {"version": VERSION, "run": str(run), "source_path": str(source), "source_video_sha256": source_id,
        "results_path": str(results_path), "results_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "page_id": page["page_id"], "document_type": final.document_type,
        "model": configuration.get("vision_model") or configuration.get("model"),
        "freeze_seconds_each": float(freeze_seconds), "inserted_seconds": len(selected) * float(freeze_seconds),
        "display_rotation_clockwise_degrees": rotation, "selected_views": selected,
        "field_support": support, "manual_ground_truth_read": False, "manual_values_accepted": False,
        "recorded_result_errors": list(final.errors),
        "semantic_validation": "model",
        "value_comparison": "literal Unicode/case/whitespace only, independent of field",
        "word_boxes_drawn": False, "moving_frame_values_drawn": False,
        "values_source": "actual per-view native document fields in results.json; matching fused recognized values",
        "frame_origin_check": "saved page frame metadata and actual model pixel-proof SHA256; no independent video re-decoding comparison"}


def _font_path() -> Path:
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if Path(candidate).is_file():
            return Path(candidate)
    raise RuntimeError("A Cyrillic DejaVu Sans or Arial font is required")


def _wrap(draw, value, font, width):
    lines, current = [], ""
    for paragraph in str(value).splitlines() or [""]:
        for word in paragraph.split():
            proposed = (current + " " + word).strip()
            if current and draw.textlength(proposed, font=font) > width:
                lines.append(current)
                current = word
            else:
                current = proposed
        lines.append(current)
        current = ""
    return lines


def _freeze_image(view, *, rotation, model, font_path, out):
    image = Image.open(view["path"]).convert("RGB")
    source_w, source_h = image.size
    if rotation:
        image = image.transpose({90: Image.Transpose.ROTATE_270,
                                 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}[rotation])
    box = (28, 108, 1016, 970)
    image.thumbnail((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    offset = (box[0] + (box[2] - box[0] - image.width) // 2,
              box[1] + (box[3] - box[1] - image.height) // 2)
    canvas = Image.new("RGB", SIZE, "#101c2d")
    canvas.paste(image, offset)
    draw = ImageDraw.Draw(canvas)
    font = lambda size: ImageFont.truetype(str(font_path), size)
    draw.text((28, 26), f"Кадр, который прочитала модель · исходное время {view['timestamp_s']:.3f} с", font=font(34), fill="white")
    draw.text((1060, 108), "Реальные результаты чтения", font=font(31), fill="#06dfb0")
    contents = []
    for size in range(30, 15, -1):
        contents = [(name, _wrap(draw, field["raw_value"], font(size), 808)) for name, field in view["fields"].items()]
        total = sum(29 + len(lines) * (size + 7) + 17 for _, lines in contents)
        if total <= 780:
            break
    else:
        raise ValueError("Literal recognized values do not fit the demo card without truncation")
    y = 164
    for name, lines in contents:
        draw.text((1060, y), LABELS.get(name, name), font=font(22), fill="#a8b8ca")
        y += 29
        for line in lines:
            if draw.textlength(line, font=font(size)) > 808:
                raise ValueError("A literal recognized token is too long to display without truncation")
            draw.text((1060, y), line, font=font(size), fill="white")
            y += size + 7
        y += 17
    draw.text((28, 1001), "Пауза на прочитанном кадре · реквизиты подтверждены двумя кадрами", font=font(24), fill="#a8b8ca")
    draw.text((28, 1036), str(model or "native image model"), font=font(20), fill="#a8b8ca")
    canvas.save(out, "PNG")
    rw, rh = (source_h, source_w) if rotation % 180 else (source_w, source_h)
    sx, sy = image.width / rw, image.height / rh
    matrix = {0: [[sx, 0., offset[0]], [0., sy, offset[1]]],
              90: [[0., -sx, source_h * sx + offset[0]], [sy, 0., offset[1]]],
              180: [[-sx, 0., source_w * sx + offset[0]], [0., -sy, source_h * sy + offset[1]]],
              270: [[0., sx, offset[0]], [-sy, 0., source_w * sy + offset[1]]]}[rotation]
    return {"path": str(out), "sha256": sha256(out), "source_to_display_pixels": [*matrix, [0., 0., 1.]],
        "source_pixel_size": [source_w, source_h], "display_pixel_size": list(SIZE),
        "display_image_offset": list(offset), "display_image_pixel_size": list(image.size),
        "rotation_clockwise_degrees": rotation, "resize": "LANCZOS", "source_crop": None}


def _probe(path):
    binary = shutil.which("ffprobe")
    if not binary:
        raise RuntimeError("ffprobe is required")
    answer = subprocess.run([binary, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True)
    return json.loads(answer.stdout)


def render(run: Path, out: Path, *, page_id=None, freeze_seconds=1.5, font_path=None) -> dict:
    # Keep the leaf name un-followed so even a dangling existing symlink fails
    # the fresh-output guard instead of silently changing its target.
    out = Path(out).expanduser()
    out = out.parent.resolve() / out.name
    provenance = out.with_suffix(".provenance.json")
    assets = out.with_name(out.stem + ".native-reading-assets")
    for destination in (out, provenance, assets):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Native demo requires fresh output/provenance/assets: {destination}")
    plan = build_plan(run, page_id=page_id, freeze_seconds=freeze_seconds)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    source = Path(plan["source_path"])
    if out == source or out.suffix.casefold() != ".mp4":
        raise ValueError("Output must be a separate MP4 file")
    source_info = _probe(source)
    duration = float(source_info["format"]["duration"])
    has_audio = any(stream["codec_type"] == "audio" for stream in source_info["streams"])
    times = [view["timestamp_s"] for view in plan["selected_views"]]
    if not all(0 < value < duration for value in times) or times != sorted(set(times)):
        raise ValueError("Reading timestamps must identify distinct positions inside the source video")
    font_path = Path(font_path).resolve() if font_path else _font_path()
    ImageFont.truetype(str(font_path), 22)  # Fail before allocating/rendering output.
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.stem}.native-stage-", dir=out.parent) as staging_name:
        staging = Path(staging_name)
        staged_assets = staging / "assets"
        staged_assets.mkdir()
        snapshots = [_freeze_image(view, rotation=plan["display_rotation_clockwise_degrees"], model=plan["model"],
                      font_path=font_path, out=staged_assets / f"freeze-{index+1}.png") for index, view in enumerate(plan["selected_views"])]
        temporary = staging / "video.mp4"
        command = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", str(source)]
        for snapshot in snapshots:
            command += ["-loop", "1", "-framerate", "30", "-t", str(freeze_seconds), "-i", snapshot["path"]]
        count = len(times) + 1
        graph = ["[0:v:0]split=" + str(count) + "".join(f"[sv{i}]" for i in range(count))]
        if has_audio:
            graph.append("[0:a:0]asplit=" + str(count) + "".join(f"[sa{i}]" for i in range(count)))
        rotation = {0: "", 90: "transpose=1,", 180: "hflip,vflip,", 270: "transpose=2,"}[plan["display_rotation_clockwise_degrees"]]
        boundaries = [0., *times, duration]
        timeline, concat_parts = [], []
        output_offset = 0.
        for index in range(count):
            start, end = boundaries[index:index+2]
            graph.append(f"[sv{index}]trim=start={start:.12f}:end={end:.12f},setpts=PTS-STARTPTS,{rotation}"
                         f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v{index}]")
            if has_audio:
                graph.append(f"[sa{index}]atrim=start={start:.12f}:end={end:.12f},asetpts=PTS-STARTPTS,"
                             f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{index}]")
            concat_parts += [f"[v{index}]"] + ([f"[a{index}]"] if has_audio else [])
            timeline.append({"kind": "source_playback", "source_start_s": start, "source_end_s": end,
                             "planned_output_start_s": output_offset, "planned_output_end_s": output_offset + end - start,
                             "speed": 1., "recognition_values_overlaid": False})
            output_offset += end - start
            if index < len(times):
                graph.append(f"[{index+1}:v:0]trim=duration={freeze_seconds:.12f},setpts=PTS-STARTPTS,setsar=1,format=yuv420p[fv{index}]")
                if has_audio:
                    graph.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={freeze_seconds:.12f},asetpts=PTS-STARTPTS[fa{index}]")
                concat_parts += [f"[fv{index}]"] + ([f"[fa{index}]"] if has_audio else [])
                timeline.append({"kind": "actual_model_read_frame_freeze", "frame_id": plan["selected_views"][index]["frame_id"],
                    "source_timestamp_s": times[index], "planned_output_start_s": output_offset,
                    "planned_output_end_s": output_offset + freeze_seconds, "recognition_values_overlaid": True,
                    "word_boxes_drawn": False})
                output_offset += freeze_seconds
        graph.append("".join(concat_parts) + f"concat=n={len(timeline)}:v=1:a={int(has_audio)}[video]" + ("[audio]" if has_audio else ""))
        command += ["-filter_complex", ";".join(graph), "-map", "[video]"]
        if has_audio:
            command += ["-map", "[audio]", "-c:a", "aac", "-b:a", "128k"]
        command += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-fps_mode", "vfr", "-movflags", "+faststart", "-metadata:s:v:0", "rotate=0", str(temporary)]
        subprocess.run(command, check=True, capture_output=True)
        actual_info = _probe(temporary)
        actual_duration = float(actual_info["format"]["duration"])
        if abs(actual_duration - output_offset) > .2:
            raise ValueError("Encoded duration differs materially from source playback plus reading pauses")
        # Recheck inputs so changed files cannot inherit already-validated proof.
        if sha256(source) != plan["source_video_sha256"] or sha256(Path(plan["results_path"])) != plan["results_sha256"]:
            raise ValueError("Source video or results changed during rendering")
        for view in plan["selected_views"]:
            if sha256(Path(view["path"])) != view["source_frame_sha256"]:
                raise ValueError("Model-read frame changed during rendering")
        manifest = {**plan, "output_path": str(out), "output_sha256": sha256(temporary), "source_duration_seconds": duration,
            "expected_output_duration_seconds": output_offset, "actual_output_duration_seconds": actual_duration,
            "duration_difference_seconds": round(actual_duration - output_offset, 9),
            "audio": "source audio preserved during playback; silence during inserted pauses" if has_audio else "source has no audio",
            "font": {"path": str(font_path), "sha256": sha256(font_path)},
            "freeze_images": [{**snapshot, "path": str(assets / Path(snapshot["path"]).name)} for snapshot in snapshots],
            "timeline": timeline, "render_filter_graph": graph,
            "encoding": {"video": "libx264", "crf": 20, "pixel_format": "yuv420p", "fps_mode": "vfr"},
            "publication": "fresh artifact paths only; complete staged provenance before publication; rollback owned files on failure"}
        staged_provenance = staging / "provenance.json"
        staged_provenance.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        published = []
        assets_created = False
        try:
            # mkdir/link fail rather than overwrite if another writer appears
            # after the initial fresh-output guard.
            assets.mkdir(exist_ok=False)
            assets_created = True
            for snapshot in snapshots:
                original = Path(snapshot["path"])
                destination = assets / original.name
                os.link(original, destination)
                published.append((destination, original))
            for original, destination in ((temporary, out), (staged_provenance, provenance)):
                os.link(original, destination)
                published.append((destination, original))
        except BaseException:
            # Do not remove a concurrent replacement or any pre-existing file.
            for destination, original in reversed(published):
                try:
                    if os.path.samestat(destination.lstat(), original.lstat()):
                        destination.unlink()
                except OSError:
                    pass
            if assets_created:
                try:
                    assets.rmdir()  # A concurrent foreign file is preserved.
                except OSError:
                    pass
            raise
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--page", default=None)
    parser.add_argument("--freeze-seconds", type=float, default=1.5)
    parser.add_argument("--font", type=Path, default=None)
    args = parser.parse_args()
    manifest = render(args.run, args.out, page_id=args.page, freeze_seconds=args.freeze_seconds, font_path=args.font)
    print(json.dumps({"video": manifest["output_path"], "provenance": str(args.out.with_suffix(".provenance.json")),
        "source_seconds": manifest["source_duration_seconds"], "inserted_seconds": manifest["inserted_seconds"],
        "actual_output_seconds": manifest["actual_output_duration_seconds"],
        "rendered_field_names": list(manifest["field_support"]), "actual_frame_ids": [v["frame_id"] for v in manifest["selected_views"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
