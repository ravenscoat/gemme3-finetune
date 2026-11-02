"""Pack Python files using Windows I/O to avoid thousands of DrvFS round trips."""
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / ".venv-wsl/lib/python3.12/site-packages"
archive = ROOT / ".tools/native-python-files.zip"
links = []
count = total = 0
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
    for directory, dirs, names in os.walk(source):
        dirs[:] = [name for name in dirs if name != "__pycache__"]
        for name in names:
            item = Path(directory) / name
            relative = item.relative_to(source).as_posix()
            size = item.stat().st_size
            if ".so" in name or size > 8*1024*1024:
                links.append(relative)
            else:
                output.write(item, relative)
                count += 1
                total += size
    output.writestr("_native_binary_links.json", json.dumps(links))
print(json.dumps({"archive": str(archive), "copied_files": count, "copied_bytes": total, "linked_binaries": len(links)}), flush=True)
