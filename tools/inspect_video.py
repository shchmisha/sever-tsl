"""Extract time-stamped diagnostic samples; this is not document recognition."""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def inspect(source: Path, out: Path, interval: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    frames = out / "sampled-frames"
    frames.mkdir(exist_ok=True)
    if list(frames.glob("frame-*.jpg")):
        raise ValueError(f"Use a fresh output directory: {frames}")
    probe = subprocess.run(
        ["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_format",
         "-show_streams", "-of", "json", str(source)],
        capture_output=True, text=True, check=True,
    )
    (out / "metadata.json").write_text(probe.stdout)
    process = subprocess.run(
        ["/opt/homebrew/bin/ffmpeg", "-hide_banner", "-i", str(source), "-an",
         "-vf", f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',scale=720:-2,showinfo",
         "-fps_mode", "vfr", "-q:v", "3", str(frames / "frame-%03d.jpg")],
        capture_output=True, text=True,
    )
    (out / "extraction.log").write_text(process.stderr)
    process.check_returncode()
    timestamps = []
    for line in process.stderr.splitlines():
        if "Parsed_showinfo" in line and "pts_time:" in line:
            match = re.search(r"n:\s*(\d+).*?pts_time:\s*([0-9.]+)", line)
            if match:
                timestamps.append((int(match.group(1)), float(match.group(2))))
    files = sorted(frames.glob("frame-*.jpg"))
    if not files or len(files) != len(timestamps):
        raise ValueError(f"Image/time mismatch: {len(files)} images, {len(timestamps)} times")
    rows, previous = [], None
    for path, (index, timestamp) in zip(files, timestamps):
        image = Image.open(path)
        gray = np.asarray(image.convert("L").resize((180, 320)), dtype=float)
        center = gray[30:285, 12:168]
        laplacian = (-4 * center[1:-1, 1:-1] + center[:-2, 1:-1]
                     + center[2:, 1:-1] + center[1:-1, :-2] + center[1:-1, 2:])
        difference = None if previous is None else float(np.mean(np.abs(gray - previous)))
        rows.append({
            "sample_id": index + 1, "timestamp_s": timestamp, "path": str(path),
            "width": image.width, "height": image.height,
            "laplacian_variance_center": round(float(laplacian.var()), 3),
            "mean_abs_diff_previous": None if difference is None else round(difference, 3),
            "near_white_fraction_center": round(float((center > 245).mean()), 5),
        })
        previous = gray
    (out / "sampling.json").write_text(json.dumps({
        "source": str(source),
        "sampling": f"Source-time select with minimum spacing {interval} s; FFmpeg applies rotation metadata. Indicators are diagnostic, not field accuracy or a calibrated quality score.",
        "samples": rows,
    }, ensure_ascii=False, indent=2))
    with (out / "frame-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    sheets = []
    for chunk_number, start in enumerate(range(0, len(rows), 24), 1):
        chunk = rows[start:start + 24]
        tile_width, tile_height, label_height, columns = 180, 320, 28, 6
        sheet = Image.new("RGB", (columns * tile_width, ((len(chunk) + columns - 1) // columns) * (tile_height + label_height)), (230, 230, 230))
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(chunk):
            thumbnail = Image.open(row["path"])
            thumbnail.thumbnail((tile_width, tile_height))
            x, y = (i % columns) * tile_width, (i // columns) * (tile_height + label_height)
            sheet.paste(thumbnail, (x + (tile_width - thumbnail.width) // 2, y))
            draw.text((x + 4, y + tile_height + 3), f"#{row['sample_id']:02d} {row['timestamp_s']:.2f}s", fill="black", font=font)
        sheet_path = out / f"contact-sheet-{chunk_number}.jpg"
        sheet.save(sheet_path, quality=92)
        sheets.append(str(sheet_path))
    return {"output": str(out), "sample_count": len(rows), "sample_dimensions": [rows[0]["width"], rows[0]["height"]], "contact_sheets": sheets}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    print(json.dumps(inspect(args.source.resolve(), args.output.resolve(), args.interval), ensure_ascii=False))
