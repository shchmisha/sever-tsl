"""Start the local research API with an ephemeral, hidden provider credential."""
import argparse
import getpass
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/jobs"))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    key = (os.getenv("HARNESS_MODEL_API_KEY") or getpass.getpass(
        "RouterAI API key (process only, not saved): ")).strip().rstrip("\\")
    if not key:
        parser.error("Provider key is required")
    os.environ["HARNESS_MODEL_API_KEY"] = key
    os.environ["SEVER_RUNS_DIR"] = str(args.runs_dir.expanduser().resolve())
    os.environ.setdefault("HARNESS_MODEL_BASE_URL", "https://routerai.ru/api/v1")
    import uvicorn
    uvicorn.run("sever.api:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
