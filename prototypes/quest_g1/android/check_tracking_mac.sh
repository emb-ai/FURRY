#!/bin/zsh
set -eu
cd -- "${0:A:h}/.."
G1_PACKAGES="$PWD/.venv/lib/python3.12/site-packages"
mkdir -p android/build outputs
clang++ -O2 -std=c++17 -Ivendor/mujoco/include -Ivendor/onnxruntime-android/headers \
  android/native/check_tracking.cpp android/native/simulation.cpp android/native/retarget.cpp \
  "$G1_PACKAGES/mujoco/libmujoco.3.3.7.dylib" \
  "$G1_PACKAGES/onnxruntime/capi/libonnxruntime.1.23.2.dylib" \
  -o android/build/check-tracking-native
DYLD_LIBRARY_PATH="$G1_PACKAGES/mujoco:$G1_PACKAGES/onnxruntime/capi" \
  android/build/check-tracking-native android/assets
clang++ -O2 -std=c++17 -Ivendor/mujoco/include -Ivendor/onnxruntime-android/headers \
  android/native/check_tracking_frames.cpp android/native/simulation.cpp android/native/retarget.cpp \
  "$G1_PACKAGES/mujoco/libmujoco.3.3.7.dylib" \
  "$G1_PACKAGES/onnxruntime/capi/libonnxruntime.1.23.2.dylib" \
  -o android/build/check-tracking-frames
DYLD_LIBRARY_PATH="$G1_PACKAGES/mujoco:$G1_PACKAGES/onnxruntime/capi" \
  android/build/check-tracking-frames android/assets
