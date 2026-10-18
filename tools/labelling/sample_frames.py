"""Sample a video every N seconds (source time), keep exact PTS, build labelled contact sheets.

Diagnostic sampling for human labelling only; it never calls a model and never
reads pipeline output. Portable: needs ffmpeg/ffprobe and Pillow.
"""
import argparse, json, re, subprocess, hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for b in iter(lambda: s.read(8 << 20), b""): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--interval", type=float, default=0.5)
    a = ap.parse_args()
    frames = a.out / "sampled-frames"; frames.mkdir(parents=True, exist_ok=False)
    meta = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(a.source)],
                          capture_output=True, text=True, check=True).stdout
    (a.out / "metadata.json").write_text(meta)
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-threads", "8", "-i", str(a.source), "-an", "-vf",
        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{a.interval})',scale=720:-2,showinfo",
        "-fps_mode", "vfr", "-q:v", "3", str(frames / "frame-%03d.jpg")], capture_output=True, text=True)
    (a.out / "extraction.log").write_text(proc.stderr); proc.check_returncode()
    times = [float(m.group(1)) for m in re.finditer(r"Parsed_showinfo.*?pts_time:\s*([0-9.]+)", proc.stderr)]
    files = sorted(frames.glob("frame-*.jpg"))
    assert len(files) == len(times), (len(files), len(times))
    rows = [{"sample_id": i + 1, "timestamp_s": t, "file": f.name} for i, (f, t) in enumerate(zip(files, times))]
    fmt = json.loads(meta)["format"]
    (a.out / "sampling.json").write_text(json.dumps({"source": str(a.source), "source_sha256": sha256(a.source),
        "duration_s": float(fmt.get("duration", 0)), "interval_s": a.interval, "samples": rows}, ensure_ascii=False, indent=1))
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError: font = ImageFont.load_default()
    tw, th, lh, cols = 240, 427, 30, 6
    for n, start in enumerate(range(0, len(rows), 24), 1):
        chunk = rows[start:start + 24]
        sheet = Image.new("RGB", (cols * tw, ((len(chunk) + cols - 1) // cols) * (th + lh)), (225, 225, 225))
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(chunk):
            im = Image.open(frames / r["file"]); im.thumbnail((tw, th))
            x, y = (i % cols) * tw, (i // cols) * (th + lh)
            sheet.paste(im, (x + (tw - im.width) // 2, y))
            d.text((x + 4, y + th + 4), f"#{r['sample_id']:03d} {r['timestamp_s']:.2f}s", fill="black", font=font)
        sheet.save(a.out / f"contact-sheet-{n:02d}.jpg", quality=90)
    print(json.dumps({"out": str(a.out), "samples": len(rows), "duration_s": float(fmt.get("duration", 0)), "sheets": n}))

main()
