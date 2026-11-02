"""Copy small Python files to ext4; link large verified binaries back to D:."""
import json
import os
from pathlib import Path
import shutil
import sys

workspace = Path("/mnt/d/gemma-finetune")
destination = Path(sys.argv[1]).resolve()
source = workspace / ".venv-wsl/lib/python3.12/site-packages"
target = destination / "venv/lib/python3.12/site-packages"
if not target.is_dir() or not destination.is_relative_to(Path.home()):
    raise ValueError("Expected a newly created task environment under the user's home")
copied = linked = copied_bytes = 0
for directory, subdirs, filenames in os.walk(source):
    subdirs[:] = [name for name in subdirs if name != "__pycache__"]
    relative = Path(directory).relative_to(source)
    output = target / relative
    output.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        item, result = Path(directory) / name, output / name
        if result.exists() or result.is_symlink():
            continue
        size = item.stat().st_size
        if ".so" in name or size > 8*1024*1024:
            result.symlink_to(item.resolve())
            linked += 1
        else:
            shutil.copy2(item, result)
            copied += 1
            copied_bytes += size
summary = {"runtime_root": str(destination), "small_files_copied": copied,
           "binary_files_linked": linked, "copied_bytes": copied_bytes,
           "binary_source": str(source)}
(workspace / "reports/native_runtime.json").write_text(json.dumps(summary, indent=2)+"\n")
(workspace / ".tools/native-python.txt").write_text(str(destination / "venv/bin/python")+"\n")
print(json.dumps(summary, indent=2), flush=True)
