"""Resume a large public file in independently checked HTTP range segments."""
import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def download(url, target, size, sha256, workers=24):
    target = Path(target)
    if target.exists() and target.stat().st_size == size:
        with target.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() == sha256:
                return
    parts = target.parent / (target.name + ".chunks")
    parts.mkdir(parents=True, exist_ok=True)
    chunk_size = 2 * 1024 * 1024
    count = math.ceil(size/chunk_size)

    def fetch_chunk(index):
        start, end = index*chunk_size, min(size, (index+1)*chunk_size)-1
        output = parts / f"{index:06d}"
        expected = end-start+1
        if output.exists() and output.stat().st_size == expected:
            return expected
        temporary = output.with_suffix(".part")
        # Distinct query keys prevent intermediary caches reusing another range.
        chunk_url = (url + ("&" if "?" in url else "?") + f"segment={index}"
                     if "huggingface.co/" in url else url)
        last_error = ""
        for attempt in range(4):
            result = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                     "--connect-timeout", "20", "--max-time", "90",
                                     "--range", f"{start}-{end}", "--output", str(temporary), chunk_url],
                                    capture_output=True, text=True)
            if result.returncode == 0 and temporary.exists() and temporary.stat().st_size == expected:
                temporary.replace(output)
                return expected
            last_error = result.stderr[-300:]
            time.sleep(attempt+1)
        raise RuntimeError(f"Segment {index} failed: {last_error}")

    completed = 0
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_chunk, i) for i in range(count)]
        for future in concurrent.futures.as_completed(futures):
            completed += future.result()
            if completed % (20*chunk_size) == 0 or completed == size:
                print(f"{target.name}: {completed/size:.1%}, {completed/1024**2:.0f} MiB; {time.time()-started:.0f}s", flush=True)
    temporary = target.with_suffix(target.suffix + ".assembling")
    with temporary.open("wb") as output:
        for index in range(count):
            with (parts / f"{index:06d}").open("rb") as source:
                while block := source.read(1024*1024):
                    output.write(block)
    with temporary.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if temporary.stat().st_size != size or digest != sha256:
        raise ValueError("Assembled download failed size or SHA-256 validation")
    temporary.replace(target)
    print("Verified", target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-directory", required=True)
    args = parser.parse_args()
    directory = Path(args.model_directory)
    info = json.loads((directory / "source_repository.json").read_text())
    for item in info["siblings"]:
        if item["rfilename"].endswith(".safetensors"):
            download(f"https://huggingface.co/{info['id']}/resolve/{info['sha']}/{item['rfilename']}",
                     directory / item["rfilename"], item["size"], item["lfs"]["sha256"])
