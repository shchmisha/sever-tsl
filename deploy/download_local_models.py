"""Download pinned GGUF files on the inference host and verify publisher hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(repo, revision, names, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with urlopen("https://huggingface.co/api/models/" + repo + "/revision/" + revision + "?blobs=true", timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != revision:
        raise ValueError("Model revision must be an immutable full commit hash")
    by_name = {item["rfilename"]: item for item in metadata["siblings"]}
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError("Unique file basenames required")

    def download(name):
        item = by_name[name]
        expected = item["lfs"]["sha256"]
        size = item["lfs"]["size"]
        path = destination / name
        if path.is_file() and path.stat().st_size == size and digest(path) == expected:
            return {"name": name, "path": str(path), "bytes": size, "sha256": expected, "status": "existing_verified"}
        partial = path.with_suffix(path.suffix + ".download")
        url = "https://huggingface.co/" + repo + "/resolve/" + revision + "/" + quote(name)
        prefix = partial.stat().st_size if partial.exists() else 0
        if size - prefix > 1024 * 1024 * 1024:
            # Separate range files preserve the verified-size prefix if a
            # transfer is interrupted; no sparse file is mistaken for progress.
            parts = path.with_suffix(path.suffix + ".parts")
            parts.mkdir(exist_ok=True)
            ranges = [(start, min(start + 512 * 1024 * 1024, size) - 1)
                      for start in range(prefix, size, 512 * 1024 * 1024)]

            def range_file(bounds):
                start, end = bounds
                piece = parts / f"{start}-{end}"
                length = end - start + 1
                for attempt in range(5):
                    have = piece.stat().st_size if piece.exists() else 0
                    if have == length:
                        return piece
                    if have > length:
                        raise ValueError("Range file exceeds expected length")
                    offset = start + have
                    request = Request(url + f"?download=true&range_start={offset}&range_end={end}",
                                      headers={"Range": f"bytes={offset}-{end}"})
                    try:
                        with urlopen(request, timeout=120) as response:
                            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-{end}/"):
                                raise ValueError("Server did not honor the requested model range")
                            with piece.open("ab") as stream:
                                while block := response.read(8 * 1024 * 1024):
                                    stream.write(block)
                        if piece.stat().st_size == length:
                            return piece
                    except OSError:
                        if attempt == 4:
                            raise
                    time.sleep(2)
                raise ValueError("Incomplete model range")

            with ThreadPoolExecutor(max_workers=8) as pool:
                downloaded = list(pool.map(range_file, ranges))
            with partial.open("ab") as stream:
                for piece in downloaded:
                    with piece.open("rb") as source:
                        while block := source.read(8 * 1024 * 1024):
                            stream.write(block)
        for attempt in range(5):
            start = partial.stat().st_size if partial.exists() else 0
            if start >= size:
                break
            request = Request(url, headers={"Range": f"bytes={start}-"} if start else {})
            try:
                with urlopen(request, timeout=120) as response:
                    if start and response.status != 206:
                        start = 0
                    if start and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                        raise ValueError("Download resumed at an unexpected offset")
                    with partial.open("ab" if start else "wb") as stream:
                        while block := response.read(8 * 1024 * 1024):
                            stream.write(block)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2)
        if not partial.exists() or partial.stat().st_size != size or digest(partial) != expected:
            raise ValueError("Model file integrity mismatch: " + name)
        partial.replace(path)
        parts = path.with_suffix(path.suffix + ".parts")
        if parts.exists():
            for piece in parts.iterdir():
                piece.unlink()
            parts.rmdir()
        row = {"name": name, "path": str(path), "bytes": size, "sha256": expected, "status": "downloaded_on_server"}
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(download, names))
    result = {"repo": repo, "revision": revision, "files": files}
    (destination / "download-provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--file", action="append", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.revision, args.file, args.destination)
