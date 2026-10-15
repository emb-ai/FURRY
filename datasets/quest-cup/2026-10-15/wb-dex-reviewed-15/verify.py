"""Verify original archives against the reviewed selection manifest."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
manifest = json.loads((root / "index.json").read_text(encoding="utf-8"))
assert len(manifest["episodes"]) == manifest["count"] == 67
for episode in manifest["episodes"]:
    archive = root / episode["archive"]
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == episode["sha256"], f"Hash mismatch: {archive.name}; check Git LFS download"
    with zipfile.ZipFile(archive) as source:
        assert source.testzip() is None, f"ZIP CRC failed: {archive.name}"
print("Verified 67 episode archives: SHA256 and ZIP CRC.")
