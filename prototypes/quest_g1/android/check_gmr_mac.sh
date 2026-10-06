#!/bin/zsh
set -eu
cd -- "${0:A:h}/.."
G1_PACKAGES="$PWD/.venv/lib/python3.12/site-packages"
export DYLD_LIBRARY_PATH="$G1_PACKAGES/mujoco:$G1_PACKAGES/onnxruntime/capi"
mkdir -p android/build outputs/gmr-recording
.venv/bin/python scripts/export_gmr.py
clang++ -O2 -std=c++17 -Ivendor/mujoco/include android/native/gmr.cpp android/native/gmr_oracle.cpp "$G1_PACKAGES/mujoco/libmujoco.3.3.7.dylib" -o android/build/gmr-oracle
.venv/bin/python scripts/check_gmr_parity.py
for G1_TOOL in check_meta_retarget check_leg_mapping replay_gmr; do
  clang++ -O2 -std=c++17 -Ivendor/mujoco/include -Ivendor/onnxruntime-android/headers \
    "android/native/$G1_TOOL.cpp" android/native/gmr.cpp android/native/meta_retarget.cpp android/native/simulation.cpp \
    "$G1_PACKAGES/mujoco/libmujoco.3.3.7.dylib" "$G1_PACKAGES/onnxruntime/capi/libonnxruntime.1.23.2.dylib" -o "android/build/$G1_TOOL"
done
android/build/check_leg_mapping android/assets
cp android/assets/recording_metadata.json outputs/gmr-recording/recording_metadata.json
android/build/check_meta_retarget android/assets outputs/gmr-recording > outputs/gmr-meta-check.log
cat outputs/gmr-meta-check.log
G1_EPISODE=$(sed -n 's/^episode=//p' outputs/gmr-meta-check.log)
android/build/replay_gmr android/assets "$G1_EPISODE"
.venv/bin/python scripts/inspect_recording.py "$G1_EPISODE"
