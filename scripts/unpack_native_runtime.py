import json
from pathlib import Path
import zipfile

ROOT = Path("/mnt/d/gemma-finetune")
runtime = Path.home() / ".local/share/gemma-finetune-20261030"
target = runtime / "venv/lib/python3.12/site-packages"
source = ROOT / ".venv-wsl/lib/python3.12/site-packages"
with zipfile.ZipFile(ROOT / ".tools/native-python-files.zip") as archive:
    for name in archive.namelist():
        if not (target / name).resolve().is_relative_to(target.resolve()):
            raise ValueError("Unsafe runtime archive entry")
    archive.extractall(target)
links = json.loads((target / "_native_binary_links.json").read_text())
for name in links:
    destination = target / name
    if not destination.exists() and not destination.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source / name)
summary = {"runtime_root": str(runtime), "binary_files_linked": len(links),
           "package_source": str(source), "method": "Windows-native archive; ext4 extraction; shared binary symlinks"}
(ROOT / "reports/native_runtime.json").write_text(json.dumps(summary, indent=2)+"\n")
(ROOT / ".tools/native-python.txt").write_text(str(runtime / "venv/bin/python")+"\n")
print(json.dumps(summary, indent=2), flush=True)
