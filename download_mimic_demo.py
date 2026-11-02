"""Download the official open MIMIC-IV demo; no third-party packages needed."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
URL = "https://physionet.org/content/mimic-iv-demo/get-zip/2.2/"
ARCHIVE = ROOT / "data" / "mimic-iv-demo-2.2.zip"
DESTINATION = ROOT / "data" / "mimic-iv-demo-2.2"


def main():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        temporary = ARCHIVE.with_suffix(".zip.part")
        with urllib.request.urlopen(URL, timeout=120) as response:
            with temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
        temporary.replace(ARCHIVE)
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Corrupt archive member: {bad_member}")
        destination = DESTINATION.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive path: {member.filename}")
        archive.extractall(destination)
    patients_files = list(DESTINATION.rglob("patients.csv")) + list(DESTINATION.rglob("patients.csv.gz"))
    if len(patients_files) != 1:
        raise ValueError("Expected exactly one patients table in the official demo")
    open_table = gzip.open if patients_files[0].suffix == ".gz" else open
    with open_table(patients_files[0], "rt", encoding="utf-8", newline="") as source:
        patients = list(csv.DictReader(source))
    if len(patients) != 100:
        raise ValueError(f"Expected 100 demo patients, found {len(patients)}")
    with ARCHIVE.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    manifest = {
        "source_url": URL,
        "source_page": "https://physionet.org/content/mimic-iv-demo/2.2/",
        "version": "2.2",
        "archive_sha256": checksum,
        "patients": len(patients),
        "csv_files": len(list(DESTINATION.rglob("*.csv"))) + len(list(DESTINATION.rglob("*.csv.gz"))),
        "purpose": "Pipeline development only; insufficient for a final population assessment",
        "license": "Open Data Commons Open Database License v1.0; see extracted LICENSE.txt",
    }
    (ARCHIVE.parent / "demo_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
