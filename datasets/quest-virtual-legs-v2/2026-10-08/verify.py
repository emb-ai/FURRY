#!/usr/bin/env python3
"""Verify this dataset release without changing or extracting any files."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile


class IntegrityError(ValueError):
    """An artifact does not match the declared dataset release."""


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def safe_path(root, relative):
    require(isinstance(relative, str) and relative, "File paths must be nonempty strings")
    parts = PurePosixPath(relative)
    require(
        not parts.is_absolute()
        and "\\" not in relative
        and ":" not in relative
        and all(part not in ("", ".", "..") for part in relative.split("/")),
        f"Unsafe relative path: {relative!r}",
    )
    path = root.joinpath(*parts.parts)
    require(path.resolve().is_relative_to(root), f"Path leaves dataset directory: {relative}")
    require(not any(p.is_symlink() for p in (path, *path.parents) if p != root.parent),
            f"Symlinks are not release payloads: {relative}")
    require(path.is_file(), f"Missing file: {relative}")
    with path.open("rb") as stream:
        prefix = stream.read(128)
    require(not prefix.startswith(b"version https://git-lfs.github.com/spec/v1"),
            f"Git LFS pointer instead of data: {relative}; run git lfs pull")
    return path


def sha256_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def read_json(root, relative):
    try:
        return json.loads(safe_path(root, relative).read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IntegrityError(f"Invalid JSON in {relative}: {error}") from error


def valid_sha(value, context):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            f"Invalid SHA256 in {context}")


def verify_release(root):
    release = read_json(root, "release.json")
    require(isinstance(release, dict), "release.json must contain an object")
    records = release.get("files")
    require(isinstance(records, list) and records, "release.json files must be a nonempty list")
    index = {}
    for record in records:
        require(isinstance(record, dict), "Release file records must be objects")
        relative = record.get("path")
        path = safe_path(root, relative)
        require(relative not in index, f"Duplicate release file: {relative}")
        expected_size = record.get("size")
        require(type(expected_size) is int and expected_size >= 0,
                f"Invalid size in release record: {relative}")
        valid_sha(record.get("sha256"), relative)
        require(path.stat().st_size == expected_size, f"Size mismatch: {relative}")
        require(sha256_file(path) == record["sha256"], f"SHA256 mismatch: {relative}")
        index[relative] = record
    return release, index


def indexed_path(root, index, relative):
    require(relative in index, f"Payload missing from release.json: {relative}")
    return safe_path(root, relative)


def verify_npz(path, clip, np):
    try:
        with np.load(path, allow_pickle=False) as data:
            expected = {"qpos", "fps"}
            body_keys = {"local_body_pos", "link_body_list"}
            require(set(data.files) in (expected, expected | body_keys),
                    f"Unexpected NPZ fields: {clip['file']}")
            require(clip["split"] == "replay" or set(data.files) == expected | body_keys,
                    f"Quest clip lacks body arrays: {clip['file']}")
            arrays = {key: data[key] for key in data.files}
            for key, array in arrays.items():
                require(array.dtype.kind != "O", f"Object array {key}: {clip['file']}")
                if array.dtype.kind in "biufc":
                    require(bool(np.isfinite(array).all()),
                            f"Non-finite array {key}: {clip['file']}")
            qpos = arrays["qpos"]
            require(qpos.ndim == 2 and qpos.shape[1] == 36 and qpos.shape[0] >= 2,
                    f"qpos must have shape (N >= 2, 36): {clip['file']}")
            require(qpos.dtype.kind == "f", f"qpos must be floating point: {clip['file']}")
            fps = arrays["fps"]
            require(fps.shape == () and fps.dtype.kind in "iuf" and float(fps) == 100.0,
                    f"fps must be scalar 100: {clip['file']}")
            norms = np.linalg.norm(qpos[:, 3:7], axis=1)
            require(bool(np.allclose(norms, 1.0, rtol=0.0, atol=1e-6)),
                    f"Root quaternion is not normalized: {clip['file']}")
            if body_keys <= arrays.keys():
                body = arrays["local_body_pos"]
                names = arrays["link_body_list"]
                require(body.shape == (len(qpos), 38, 3) and body.dtype.kind == "f",
                        f"local_body_pos must have shape (N, 38, 3): {clip['file']}")
                require(names.shape == (38,) and names.dtype.kind in "US",
                        f"link_body_list must contain 38 names: {clip['file']}")
                require(len(set(names.tolist())) == 38 and all(len(name) for name in names.tolist()),
                        f"Body names must be unique and nonempty: {clip['file']}")
            seconds = (len(qpos) - 1) / float(fps)
            require(abs(seconds - float(clip["seconds"])) < 1e-7,
                    f"Clip duration mismatch: {clip['file']}")
            return seconds
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        if isinstance(error, IntegrityError):
            raise
        raise IntegrityError(f"Invalid NPZ {clip['file']}: {error}") from error


def verify_manifest(root, index, folder, expected_counts, np):
    relative = f"{folder}/manifest.json"
    indexed_path(root, index, relative)
    manifest = read_json(root, relative)
    require(manifest.get("schema") == "quest-g1-motion-v1", f"Unexpected schema: {relative}")
    require(manifest.get("fps") == 100, f"Manifest fps must be 100: {relative}")
    require(manifest.get("validation_stages") == [2, 9, 14],
            f"Validation stages must be 2, 9, 14: {relative}")
    require(manifest.get("split_unit") == "whole protocol stage before chunking",
            f"Whole-stage split is not declared: {relative}")
    clips = manifest.get("clips")
    require(isinstance(clips, list), f"Manifest clips must be a list: {relative}")
    counts = Counter(clip.get("split") for clip in clips)
    require(dict(counts) == expected_counts, f"Unexpected clip counts in {folder}: {dict(counts)}")
    by_id, by_file = {}, set()
    seconds = Counter()
    for clip in clips:
        clip_id, file = clip["id"], clip["file"]
        require(isinstance(clip_id, str) and clip_id and clip_id not in by_id,
                f"Missing or duplicate clip id in {folder}: {clip_id}")
        require(file not in by_file and file.endswith(".npz"),
                f"Missing or duplicate NPZ path in {folder}: {file}")
        split, stage = clip["split"], clip["stage"]
        require(file.startswith(f"{split}/"), f"NPZ path disagrees with split: {file}")
        require(type(stage) is int, f"Stage must be an integer: {file}")
        if split == "replay":
            require(stage == 0, f"Public replay must have stage 0: {file}")
        else:
            require(1 <= stage <= 15, f"Invalid Quest protocol stage: {file}")
            require((stage in {2, 9, 14}) == (split == "validation"),
                    f"Whole-stage split leakage: {file}")
        payload = f"{folder}/{file}"
        path = indexed_path(root, index, payload)
        valid_sha(clip.get("sha256"), payload)
        require(index[payload]["sha256"] == clip["sha256"],
                f"Manifest and release SHA256 disagree: {payload}")
        seconds[split] += verify_npz(path, clip, np)
        by_id[clip_id] = clip
        by_file.add(file)
    declared_seconds = manifest.get("duration_seconds", {})
    require("train" in declared_seconds and "validation" in declared_seconds,
            f"Missing train/validation duration: {folder}")
    for split, duration in declared_seconds.items():
        require(split in seconds and abs(seconds[split] - float(duration)) < 1e-6,
                f"Manifest {split} duration mismatch: {folder}")
    for clip in clips:
        parent_id = clip.get("parent_id")
        if parent_id is not None:
            require(clip["split"] == "train" and parent_id in by_id,
                    f"Missing train-only mirror parent: {clip['id']}")
            parent = by_id[parent_id]
            require(parent["split"] == "train" and "parent_id" not in parent,
                    f"Mirror parent split leakage: {clip['id']}")
            require(clip.get("augmentation") == "sagittal_reflection",
                    f"Unexpected augmentation: {clip['id']}")
            for field in ("stage", "seconds", "calibration_sequence",
                          "source_start_wall_s", "source_end_wall_s"):
                require(clip.get(field) == parent.get(field),
                        f"Mirror changes parent source field {field}: {clip['id']}")
        else:
            require("augmentation" not in clip, f"Augmentation lacks parent_id: {clip['id']}")
    return manifest, by_id, {split: round(value, 8) for split, value in seconds.items()}


def verify_source_zip(root, index, release, manifest):
    relative = release.get("source_zip", "raw/source-episode.zip")
    path = indexed_path(root, index, relative)
    source = manifest["source_episode"]
    require(isinstance(source, str) and re.fullmatch(r"episode-[0-9]+", source),
            "Invalid source episode identifier")
    expected = manifest.get("source_sha256")
    require(isinstance(expected, dict) and set(expected) == {
        "input.csv", "body.csv", "capture_timeline.csv", "capture_plan.json"
    }, "Expected four source file SHA256 records")
    try:
        with zipfile.ZipFile(path) as archive:
            seen = set()
            for member in archive.infolist():
                name = member.filename
                trimmed = name[:-1] if member.is_dir() else name
                parts = PurePosixPath(trimmed)
                require(not parts.is_absolute() and "\\" not in name and ":" not in name
                        and all(part not in ("", ".", "..") for part in trimmed.split("/")),
                        f"Unsafe ZIP member: {name}")
                require(parts.parts[0] == source, f"Unexpected ZIP source directory: {name}")
                require(name not in seen, f"Duplicate ZIP member: {name}")
                require(not stat.S_ISLNK(member.external_attr >> 16), f"ZIP symlink: {name}")
                seen.add(name)
            bad_member = archive.testzip()
            require(bad_member is None, f"ZIP CRC failure: {bad_member}")
            for name, expected_sha in expected.items():
                valid_sha(expected_sha, name)
                with archive.open(f"{source}/{name}") as stream:
                    require(sha256_stream(stream) == expected_sha, f"Raw source SHA256 mismatch: {name}")
    except (OSError, KeyError, zipfile.BadZipFile, RuntimeError) as error:
        raise IntegrityError(f"Invalid source ZIP: {error}") from error
    return source


def verify(root):
    try:
        import numpy as np
    except ImportError as error:
        raise IntegrityError("NumPy is required; install numpy in your Python environment") from error
    release, index = verify_release(root)
    v2, v2_clips, v2_seconds = verify_manifest(
        root, index, "dataset-v2", {"train": 33, "validation": 12, "replay": 4}, np
    )
    v3, v3_clips, v3_seconds = verify_manifest(
        root, index, "night-v3", {"train": 66, "validation": 12, "replay": 4}, np
    )
    require(v3["source_episode"] == v2["source_episode"]
            and v3["source_sha256"] == v2["source_sha256"], "V2/V3 source provenance differs")
    require(set(v2_clips) <= set(v3_clips), "V3 omits original V2 clips")
    for clip_id, original in v2_clips.items():
        require(v3_clips[clip_id] == original, f"V3 changes original V2 clip: {clip_id}")
    mirrors = [clip for clip in v3_clips.values() if "parent_id" in clip]
    require(len(mirrors) == 33 and not any("parent_id" in clip for clip in v2_clips.values()),
            "Expected 33 V3 mirrors and no V2 mirrors")
    require(set(clip["parent_id"] for clip in mirrors) == {
        clip["id"] for clip in v2_clips.values() if clip["split"] == "train"
    }, "Every original train clip must have exactly one mirror")
    augmentation = v3.get("augmentation", {})
    require(augmentation.get("augmented_clips") == 33
            and augmentation.get("independent_demonstrations_added") == 0,
            "Mirror augmentation counters are incorrect")
    require(augmentation.get("source_manifest_sha256") == index["dataset-v2/manifest.json"]["sha256"],
            "Augmentation source manifest SHA256 mismatch")
    indexed_path(root, index, "mirror-check.json")
    require(read_json(root, "mirror-check.json") == augmentation,
            "mirror-check.json differs from manifest augmentation checks")
    checks = augmentation.get("checks", [])
    require(len(checks) == 33 and {check.get("parent_id") for check in checks} == {
        clip["parent_id"] for clip in mirrors
    }, "Mirror checks must cover all 33 parents")
    for check in checks:
        require(0 <= float(check["max_fk_position_error_m"]) < 1e-4
                and 0 <= float(check["max_fk_rotation_matrix_error"]) < 1e-4,
                f"Mirror FK check exceeds tolerance: {check['parent_id']}")
    source = verify_source_zip(root, index, release, v2)
    return {
        "ok": True,
        "source_episode": source,
        "release_files": len(index),
        "dataset_v2": {"clips": 49, "counts": {"train": 33, "validation": 12, "replay": 4},
                       "seconds": v2_seconds},
        "night_v3": {"clips": 82, "counts": {"train": 66, "validation": 12, "replay": 4},
                     "seconds": v3_seconds, "mirrors": 33},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent,
                        help="Dataset directory (default: directory containing this script)")
    args = parser.parse_args()
    try:
        result = verify(args.directory.resolve())
    except (IntegrityError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
