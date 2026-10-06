#!/bin/zsh
set -eu
cd -- "${0:A:h}/.."
G1_PACKAGES="$PWD/.venv/lib/python3.12/site-packages"
export DYLD_LIBRARY_PATH="$G1_PACKAGES/mujoco:$G1_PACKAGES/onnxruntime/capi"
mkdir -p android/build
clang++ -O2 -std=c++17 -Ivendor/mujoco/include -Ivendor/onnxruntime-android/headers \
  android/native/check_recording.cpp android/native/simulation.cpp android/native/retarget.cpp \
  "$G1_PACKAGES/mujoco/libmujoco.3.3.7.dylib" "$G1_PACKAGES/onnxruntime/capi/libonnxruntime.1.23.2.dylib" \
  -o android/build/check-recording
clang++ -O2 -std=c++17 -Ivendor/mujoco/include android/native/replay_human.cpp android/native/retarget.cpp \
  "$G1_PACKAGES/mujoco/libmujoco.3.3.7.dylib" -o android/build/replay-human
mkdir -p outputs/recording-fixture
cp android/assets/recording_metadata.json outputs/recording-fixture/recording_metadata.json
# This fixture intentionally exercises the retained legacy wrist-only replay.
.venv/bin/python -c 'import json,pathlib; p=pathlib.Path("outputs/recording-fixture/recording_metadata.json"); d=json.loads(p.read_text()); d["retargeting"]="legacy_wrist_fixture"; p.write_text(json.dumps(d))'
G1_EPISODE=$(android/build/check-recording android/assets outputs/recording-fixture)
print "$G1_EPISODE"
android/build/replay-human android/assets "$G1_EPISODE"
.venv/bin/python scripts/inspect_recording.py "$G1_EPISODE"
