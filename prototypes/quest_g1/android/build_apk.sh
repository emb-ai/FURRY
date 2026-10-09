#!/bin/zsh
set -eu
cd -- "${0:A:h}/.."
G1_PROJECT="$PWD"
export JAVA_HOME="$G1_PROJECT/.tools/jdk/Contents/Home"
G1_SDK="$G1_PROJECT/.tools/android-sdk"
G1_NDK="$G1_SDK/ndk/27.2.12479018"
G1_BT="$G1_SDK/build-tools/android-14"
if [[ ! -f vendor/IXWebSocket/ixwebsocket/IXWebSocket.h ]]; then
  print "Missing camera transport dependency: run scripts/fetch_camera_transport.sh"
  exit 1
fi
.venv/bin/python scripts/prepare_android.py
cmake -S android -B android/build/arm64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$G1_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
  -DANDROID_STL=c++_shared -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DG1_ENABLE_FOOT_FLOOR_GUARD="${G1_FOOT_FLOOR_GUARD:-OFF}" \
  -DG1_SWING_CLEARANCE_MM="${G1_SWING_CLEARANCE_MM:-0}" \
  -DG1_ENABLE_TWIST_GROUNDING="${G1_TWIST_GROUNDING:-OFF}"
cmake --build android/build/arm64 --target g1_quest -j 6
mkdir -p android/build/package/lib/arm64-v8a
cp android/build/arm64/libg1_quest.so android/build/package/lib/arm64-v8a/
cp android/build/arm64/lib/libmujoco.so android/build/package/lib/arm64-v8a/
cp vendor/onnxruntime-android/jni/arm64-v8a/libonnxruntime.so android/build/package/lib/arm64-v8a/
cp vendor/openxr-android/jni/arm64-v8a/libopenxr_loader.so android/build/package/lib/arm64-v8a/
cp "$G1_NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" android/build/package/lib/arm64-v8a/
"$G1_NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-strip" --strip-unneeded android/build/package/lib/arm64-v8a/*.so
"$G1_BT/aapt2" link --manifest android/AndroidManifest.xml -I "$G1_SDK/platforms/android-34/android.jar" -A android/assets -o android/build/unsigned.apk
(cd android/build/package && /usr/bin/zip -q -r ../unsigned.apk lib)
"$G1_BT/zipalign" -f -p 4 android/build/unsigned.apk android/build/aligned.apk
if [[ ! -f .tools/g1-debug.keystore ]]; then
  "$JAVA_HOME/bin/keytool" -genkeypair -keystore .tools/g1-debug.keystore -storepass android -keypass android -alias androiddebugkey -dname 'CN=G1 Quest Lab Development' -keyalg RSA -keysize 2048 -validity 10000
fi
"$G1_BT/apksigner" sign --ks .tools/g1-debug.keystore --ks-pass pass:android --key-pass pass:android --out android/build/g1-quest.apk android/build/aligned.apk
"$G1_BT/apksigner" verify android/build/g1-quest.apk
print "APK: $G1_PROJECT/android/build/g1-quest.apk"
