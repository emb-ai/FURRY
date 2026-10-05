#!/bin/zsh
set -eu
cd -- "${0:A:h}/.."
G1_ADB="$PWD/.tools/platform-tools/adb"
"$G1_ADB" install -r android/build/g1-quest.apk
"$G1_ADB" shell am force-stop org.g1lab.quest
"$G1_ADB" shell am start -n org.g1lab.quest/android.app.NativeActivity
