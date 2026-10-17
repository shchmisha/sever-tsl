"""Render a portable demonstration from an actual, independently checked run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size: int, bold: bool = False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def wrap(draw, text, face, width):
    lines = []
    current = ""
    for word in text.split():
        proposed = f"{current} {word}".strip()
        if current and draw.textlength(proposed, font=face) > width:
            lines.append(current)
            current = word
        else:
            current = proposed
    return lines + ([current] if current else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("runs/demo-send"))
    args = parser.parse_args()
    directory = args.directory.resolve()
    run = directory / "processing/input-3s"
    result = json.loads((run / "results.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    assert validation["passed"] and validation["case_checks"]["required_fields_correct"] == 6
    assert validation["inputs"]["results_sha256"] == hashlib.sha256((run / "results.json").read_bytes()).hexdigest()
    page = result["pages"][0]
    assert page["result"]["document_type"] == "upd"
    assert not page["result"]["errors"]
    fields = page["result"]["fields"]
    assert len(fields) == 6 and all(v["status"] == "recognized" for v in fields.values())
    frame = next(f for f in page["frames"] if f["frame_id"] == page["selected_frame_id"])
    ocr = json.loads((run / "ocr/page-0001.json").read_text())
    observation = next(o for o in ocr if o["processing"]["source_path"] == frame["path"])
    oriented = Image.open(observation["processing"]["ocr_image_path"]).convert("RGB")
    clip = directory / "input-3s.mp4"
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json", str(clip)
    ]))
    duration = float(probe["format"]["duration"])

    canvas = Image.new("RGB", (1920, 1080), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    ink, muted, accent = "#152d46", "#5a6a7d", "#087855"
    draw.text((64, 42), "Север ТСЛ — распознавание из видео", font=font(52, True), fill=ink)
    draw.text((66, 113), "Пример обработки короткого фрагмента реальной съёмки", font=font(29), fill=muted)
    draw.rounded_rectangle((980, 174, 1856, 952), radius=22, fill="white")
    draw.text((64, 190), "Исходное видео → выбранный кадр", font=font(27, True), fill=ink)
    fitted = ImageOps.contain(oriented, (864, 486))
    canvas.paste(fitted, (64 + (864 - fitted.width) // 2, 242 + (486 - fitted.height) // 2))
    draw.text((64, 754), f"Кадр выбран автоматически: {frame['timestamp_s']:.2f} с фрагмента".replace(".", ","), font=font(25), fill=muted)
    draw.rounded_rectangle((64, 818, 928, 950), radius=18, fill="#e2f4ec")
    draw.text((88, 839), "Универсальный передаточный документ", font=font(31, True), fill=accent)
    draw.text((88, 893), "Тип определён · 6 обязательных полей извлечены", font=font(27), fill=accent)

    def value(key):
        return fields[key]["raw_value"] or fields[key]["normalized_value"]

    def block(label, text, x, y, size=29, width=816):
        draw.text((x, y), label, font=font(23), fill=muted)
        face = font(size, True)
        for i, line in enumerate(wrap(draw, text, face, width)):
            draw.text((x, y + 35 + i * (size + 9)), line, font=face, fill=ink)

    block("Номер счёт-фактуры", value("invoice_number"), 1012, 205, 38, 380)
    date = fields["invoice_date"]["normalized_value"].split("-")
    block("Дата", ".".join(reversed(date)), 1450, 205, 36, 350)
    block("Грузоотправитель", value("consignor"), 1012, 315, 31)
    block("Адрес грузоотправителя", value("consignor_address"), 1012, 411, 27)
    block("Грузополучатель", value("consignee"), 1012, 564, 31)
    block("Адрес грузополучателя", value("consignee_address"), 1012, 660, 27)
    draw.text((1012, 885), "6/6 полей совпали с ручной разметкой", font=font(27, True), fill=accent)
    draw.text((64, 995), f"Реальная обработка: {result['elapsed_seconds']:.0f} с · Демонстрационный ролик смонтирован · Проверен один пример", font=font(24), fill=muted)
    card = directory / "result.png"
    canvas.save(card)

    rotation = observation["processing"]["content_rotation_clockwise_degrees"]
    orientation_filter = {0: "null", 90: "transpose=1", 180: "hflip,vflip", 270: "transpose=2"}[rotation]
    filter_graph = (
        f"[1:v]{orientation_filter},scale=864:486:force_original_aspect_ratio=decrease,"
        "pad=864:486:(ow-iw)/2:(oh-ih)/2:color=0xf3f6fa,setsar=1,fps=30[v];"
        f"[0:v][v]overlay=64:242:eof_action=pass:enable='lt(t,{duration})',format=yuv420p[out]"
    )
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", "30",
        "-i", str(card), "-i", str(clip), "-filter_complex", filter_graph, "-map", "[out]",
        "-t", str(round(duration + 5, 3)), "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(directory / "demo.mp4")
    ], check=True)
    compact = {
        "document_type": "upd", "fields": fields, "source_clip": "input-3s.mp4",
        "source_clip_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
        "source_clip_duration_seconds": duration, "original_video": "20260818-4_132351.mp4",
        "original_clip_start_seconds": 1.2, "selected_clip_timestamp_seconds": frame["timestamp_s"],
        "sharing_clip": "clip-for-sharing.mp4", "sharing_clip_note": "Same cut, rotated upright and reduced to 1080p for forwarding; recognition used the 4K input-3s.mp4.",
        "processing_seconds": result["elapsed_seconds"], "usage": result["usage"],
        "independent_validation": validation, "scope": "One selected demonstration example; not population accuracy."
    }
    (directory / "result.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n")
    shutil.copyfile(run / "fields.csv", directory / "fields.csv")
    (directory / "caption.txt").write_text(
        "Демонстрация Север ТСЛ: из короткого видео система сама выбрала кадр, определила УПД и извлекла номер, дату, грузоотправителя, грузополучателя и их адреса. Все 6 полей этого примера совпали с ручной проверкой. В ролике показан результат реальной обработки; время обработки — около 26 секунд.\n"
    )
    print(json.dumps({"demo": str(directory / "demo.mp4"), "input_duration_s": duration,
                      "processing_seconds": result["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
