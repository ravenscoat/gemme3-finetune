"""Pin and download the public source, retaining its card and checksums."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "bitext" / "raw"
REPO = "bitext/Bitext-insurance-llm-chatbot-training-dataset"


def fetch(url, target):
    part = target.with_suffix(target.suffix + ".part")
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "3", "--connect-timeout", "20", "--max-time", "1200",
                    "--output", str(part), url], check=True)
    part.replace(target)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    metadata = DEST / "repository.json"
    if not metadata.exists():
        fetch(f"https://huggingface.co/api/datasets/{REPO}", metadata)
    info = json.loads(metadata.read_text(encoding="utf-8"))
    revision = info["sha"]
    names = [item["rfilename"] for item in info["siblings"]]
    selected = [name for name in names if name.endswith(".csv") or name in ("README.md", "LICENSE", "LICENSE.txt")]
    if not any(name.endswith(".csv") for name in selected):
        raise RuntimeError(f"No CSV files found in {names}")
    files = []
    for name in selected:
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"Downloading {name}", flush=True)
            fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{name}", target)
        with target.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        files.append({"path": name, "bytes": target.stat().st_size, "sha256": digest})
    manifest = {"repository": REPO, "revision": revision, "files": files,
                "source": f"https://huggingface.co/datasets/{REPO}",
                "license": "cdla-sharing-1.0; see retained dataset card"}
    (DEST.parent / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
