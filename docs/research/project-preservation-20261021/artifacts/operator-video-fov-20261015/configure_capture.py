#!/usr/bin/env python3
"""Reapply the Quest capture override after reboot, or restore pre-investigation values.

This sets compositor properties, not the OpenXR projection submitted by the app.
Property readback is NOT a verification of video framing. Use the operator recorder
and compare the fixed HUD against a baseline recording after each firmware change.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROPERTIES = {"debug.oculus.captureFovDegrees": "110", "debug.oculus.headsetOverride": "1"}
ORIGINAL = dict.fromkeys(PROPERTIES, "")
EYE_PROPERTIES = ["debug.oculus.eyeFov" + side for side in ("Up", "Down", "Inward", "Outward")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "apply", "restore"))
    parser.add_argument("--serial", default="2G0YC5ZG7T05SR")
    parser.add_argument("--wake", action="store_true", help="Wake the headset; does not reboot or stop the app")
    args = parser.parse_args()
    adb = ROOT / ".tools/platform-tools/adb"

    def shell(*words):
        return subprocess.check_output([str(adb), "-s", args.serial, "shell", shlex.join(words)], text=True).strip()

    before = {key: shell("getprop", key) for key in (*PROPERTIES, *EYE_PROPERTIES)}
    if args.mode == "apply":
        if any(before[key] for key in EYE_PROPERTIES):
            raise SystemExit("Existing eye FOV overrides found: inspect before enabling the global override gate")
        projection = shell("dumpsys", "media_projection")
        if "Media Projection:" not in projection or projection.rsplit("Media Projection:", 1)[1].strip() != "null":
            raise SystemExit("Stop the current video recording before applying capture settings")
    if args.mode != "status":
        settings = PROPERTIES if args.mode == "apply" else ORIGINAL
        for key, value in settings.items():
            shell("setprop", key, value)
        for key, value in settings.items():
            if shell("getprop", key) != value:
                raise SystemExit("Property readback failed: " + key)
    if args.wake:
        shell("input", "keyevent", "KEYCODE_WAKEUP")
    print(json.dumps({"before": before, "after": {key: shell("getprop", key) for key in PROPERTIES},
                      "framing_verified": False, "note": "Properties reset on reboot; verify an operator MP4."}, indent=2))


if __name__ == "__main__":
    main()
