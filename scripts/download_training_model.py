"""Download a pinned training-compatible model through the Windows connection."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from download_bitext import fetch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="unsloth/gemma-3-4b-it-unsloth-bnb-4bit")
    parser.add_argument("--name", default="gemma3-4b-training")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    destination = ROOT / ".cache/models" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    metadata = destination / "source_repository.json"
    if not metadata.exists():
        fetch(f"https://huggingface.co/api/models/{args.repo}?blobs=true", metadata)
    info = json.loads(metadata.read_text())
    files = []
    for item in info["siblings"]:
        name = item["rfilename"]
        if args.metadata_only and name.endswith(".safetensors"):
            continue
        if not name.endswith((".json", ".safetensors", ".model", ".jinja", ".txt", ".md")):
            continue
        target = (destination / name).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ValueError("Unsafe repository filename")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"Downloading {name} ({item.get('size', '?')} bytes)", flush=True)
            fetch(f"https://huggingface.co/{args.repo}/resolve/{info['sha']}/{quote(name)}", target)
        with target.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        expected = item.get("lfs", {}).get("sha256")
        if expected and digest != expected:
            raise ValueError(f"LFS checksum mismatch: {name}")
        files.append({"path": name, "sha256": digest, "bytes": target.stat().st_size})
    manifest = "metadata_manifest.json" if args.metadata_only else "source_manifest.json"
    (destination / manifest).write_text(json.dumps({"repo": args.repo, "revision": info["sha"], "files": files}, indent=2)+"\n")
    print("Metadata ready:" if args.metadata_only else "Model ready:", destination, flush=True)


if __name__ == "__main__":
    main()
