"""Download locked Linux wheels with resumable ranges and SHA-256 checks."""
import concurrent.futures
from pathlib import Path
import re
import subprocess
import tomllib
import json
import urllib.request
from urllib.parse import unquote, urlparse
from download_chunks import download
from pip._vendor.packaging.tags import cpython_tags, compatible_tags
from pip._vendor.packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = tomllib.loads((ROOT / "reports/pylock.linux.toml").read_text())
    destination = ROOT / ".tools/wheels"
    destination.mkdir(parents=True, exist_ok=True)
    platforms = [f"manylinux_2_{i}_x86_64" for i in range(28, 4, -1)] + ["manylinux2014_x86_64", "manylinux2010_x86_64", "linux_x86_64"]
    tags = list(cpython_tags((3, 12), abis=["cp312"], platforms=platforms))
    tags += list(compatible_tags((3, 12), interpreter="cp312", platforms=platforms))
    rank = {tag: i for i, tag in enumerate(tags)}

    def fetch_package(package):
        wheels = package.get("wheels", [])
        candidates = []
        for wheel in wheels:
            filename = unquote(Path(urlparse(wheel["url"]).path).name)
            available = parse_wheel_filename(filename)[3] & rank.keys()
            if available:
                candidates.append((min(rank[t] for t in available), wheel))
        if not candidates:
            raise RuntimeError(f"No compatible Linux CPython 3.12 wheel for {package['name']}")
        wheel = min(candidates, key=lambda pair: pair[0])[1]
        size = wheel.get("size")
        if size is None:
            response = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                       "--head", "--max-time", "60", wheel["url"]],
                                      capture_output=True, text=True, check=True)
            sizes = re.findall(r"content-length:\s*(\d+)", response.stdout, re.I)
            if not sizes:
                raise RuntimeError(f"No file size for {package['name']}")
            size = int(sizes[-1])
        filename = unquote(Path(urlparse(wheel["url"]).path).name)
        url = wheel["url"]
        if urlparse(url).hostname == "pypi.nvidia.com":
            try:
                with urllib.request.urlopen(f"https://pypi.org/pypi/{package['name']}/{package['version']}/json", timeout=20) as response:
                    metadata = json.load(response)
                matches = [f for f in metadata["urls"] if f["filename"] == filename and f["digests"]["sha256"] == wheel["hashes"]["sha256"]]
                if matches:
                    url = matches[0]["url"]
            except Exception:
                pass
        # This public mirror was tested faster; original published hashes remain
        # authoritative. A failed mirror transfer can resume from the source CDN.
        mirror = url.replace("files.pythonhosted.org", "pypi.tuna.tsinghua.edu.cn")
        try:
            download(mirror, destination / filename, size, wheel["hashes"]["sha256"], workers=4)
        except Exception:
            if mirror == wheel["url"]:
                raise
            download(wheel["url"], destination / filename, size, wheel["hashes"]["sha256"], workers=4)
        return filename

    # Small wheels first, then multiple large wheels, with bounded total workers.
    packages = sorted(lock["packages"], key=lambda p: p["wheels"][0].get("size", 10**10))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_package, p) for p in packages]
        errors = []
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                print(f"Wheel {i}/{len(packages)} ready: {future.result()}", flush=True)
            except Exception as error:
                errors.append(str(error))
                print(f"Wheel error: {error}", flush=True)
        if errors:
            raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
