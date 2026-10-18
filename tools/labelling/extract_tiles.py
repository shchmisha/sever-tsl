"""Extract native-resolution frames at given sample times, rotate upright, cut overlapping reading tiles.

For human-style labelling only (no model calls, no pipeline output). Input JSON:
[{"video": "<id>", "leaf": "<key>", "rotation": 90, "samples": [{"sample_id": 10, "timestamp_s": 4.8}]}]
"""
import json, re, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/srv/sever/validation/labelling-20261018")
TRANSPOSE = {0: None, 90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}

def grid(width, height):
    cols, rows = (4, 3) if width >= height else (3, 4)
    overlap = 0.15
    tw = round(width / (cols - (cols - 1) * overlap)); th = round(height / (rows - (rows - 1) * overlap))
    xs = [round(i * (width - tw) / (cols - 1)) for i in range(cols)]
    ys = [round(j * (height - th) / (rows - 1)) for j in range(rows)]
    return [(r, c, xs[c], ys[r], tw, th) for r in range(rows) for c in range(cols)]

def main(spec_path):
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except OSError: font = ImageFont.load_default()
    done = []
    for item in json.loads(Path(spec_path).read_text()):
        video = Path(f"/srv/sever/input/{item['video']}.mp4")
        for sample in item["samples"]:
            out = ROOT / item["video"] / "tiles" / item["leaf"] / f"s{sample['sample_id']:03d}"
            if (out / "index.json").exists():
                done.append(str(out)); continue
            out.mkdir(parents=True, exist_ok=True)
            native = out / "native.png"
            proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-ss", f"{sample['timestamp_s']:.6f}", "-i", str(video),
                                   "-an", "-vf", "showinfo", "-frames:v", "1", str(native)], capture_output=True, text=True)
            proc.check_returncode()
            with Image.open(native) as im:
                im = im.convert("RGB")
                if TRANSPOSE[item["rotation"]]: im = im.transpose(TRANSPOSE[item["rotation"]])
                pass
                w, h = im.size
                tiles = []
                for r, c, x, y, tw, th in grid(w, h):
                    name = f"r{r}c{c}.jpg"
                    im.crop((x, y, x + tw, y + th)).save(out / name, quality=93, subsampling=0)
                    tiles.append({"tile": name, "row": r, "col": c, "box_xywh": [x, y, tw, th]})
                ov = im.copy(); d = ImageDraw.Draw(ov)
                for t in tiles:
                    x, y, tw, th = t["box_xywh"]
                    d.rectangle((x, y, x + tw, y + th), outline=(220, 0, 0), width=6)
                    d.text((x + 20, y + 20), t["tile"][:-4], fill=(220, 0, 0), font=font)
                ov.thumbnail((1568, 1568)); ov.save(out / "overview-grid.jpg", quality=90)
            native.unlink()
            (out / "index.json").write_text(json.dumps({"video": item["video"], "leaf": item["leaf"], "sample_id": sample["sample_id"],
                "requested_timestamp_s": sample["timestamp_s"], "rotation_clockwise": item["rotation"], "upright_size": [w, h],
                "tiles": tiles, "note": "native-resolution crops of the upright frame; overview-grid.jpg shows tile positions"}, indent=1))
            done.append(str(out))
    print(json.dumps({"tile_sets": len(done)}))

main(sys.argv[1])
