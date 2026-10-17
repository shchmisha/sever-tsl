"""Fresh blind image trial with source-pixel zoom tools; no historical answers."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.read_reference_images import arguments, run
from sever.zoom_image_reader import read_frame_zoom


if __name__ == '__main__':
    args = arguments()
    result = asyncio.run(run(args, reader=read_frame_zoom))
    print(json.dumps({'status': result['status'], 'summary': result['summary'],
                      'usage': result['usage']}, ensure_ascii=False))
