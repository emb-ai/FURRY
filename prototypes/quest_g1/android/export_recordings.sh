#!/bin/zsh
set -eu
cd -- "${0:A:h}/.."
mkdir -p outputs
G1_STAMP=$(date +%Y%m%d-%H%M%S)
G1_ARCHIVE="outputs/quest-recordings-$G1_STAMP.tar"
.tools/platform-tools/adb exec-out run-as org.g1lab.quest tar -C files/g1 -cf - recordings > "$G1_ARCHIVE"
tar -tf "$G1_ARCHIVE" >/dev/null
print "Saved locally: $G1_ARCHIVE"
print 'Finish recording with Y before exporting. Archive is excluded from Git.'
