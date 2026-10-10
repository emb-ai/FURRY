"""Connect this Mac to the authorised Quest; optionally update an explicit APK."""
import argparse
from datetime import datetime, timezone
import ipaddress
import json
from pathlib import Path
import re
import socket
import shutil
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ADB = Path(os.environ.get("G1_ADB") or
           (str(ROOT / ".tools/platform-tools/adb")
            if (ROOT / ".tools/platform-tools/adb").is_file()
            else shutil.which("adb") or shutil.which("adb.exe") or "adb"))
STATE = ROOT / "outputs/quest-wifi/connection.json"
SERIAL = "2G0YC5ZG7T05SR"


def adb(*args, timeout=15, required=True):
    try:
        result = subprocess.run([str(ADB), *args], capture_output=True,
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if required:
            raise RuntimeError(f"ADB не ответил за {timeout} с: {' '.join(args)}") from exc
        return ""
    if required and result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def connect(ip, serial):
    target = f"{ip}:5555"
    try:
        with socket.create_connection((ip, 5555), timeout=1):
            pass
    except OSError:
        return None
    adb("connect", target, required=False)
    actual = adb("-s", target, "shell", "getprop", "ro.serialno", required=False)
    if not actual:
        # Authentication can finish shortly after `adb connect` returns.
        for _ in range(5):
            time.sleep(0.5)
            actual = adb("-s", target, "shell", "getprop", "ro.serialno", required=False)
            if actual:
                break
    if actual == serial:
        return target
    if actual:
        adb("disconnect", target, required=False)
        raise RuntimeError(f"По адресу {target} другой шлем ({actual}); операция остановлена.")
    status = adb("devices", required=False)
    if re.search(rf"^{re.escape(target)}\s+unauthorized\b", status, re.MULTILINE):
        raise RuntimeError("Quest запросил авторизацию по Wi-Fi. В шлеме подтвердите Allow debugging / "
                           "Always allow from this computer и повторите команду. Подключение оставлено открытым.")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("apk", nargs="?", type=Path,
                        help="Установить указанный APK через install -r после подключения")
    parser.add_argument("--ip", type=ipaddress.IPv4Address, help="IP, если он изменился без USB")
    parser.add_argument("--wifi-only", action="store_true", help="Подключиться без включения ADB через USB")
    parser.add_argument("--serial", default=SERIAL, help="USB serial Quest (по умолчанию общий шлем команды)")
    args = parser.parse_args()
    serial = args.serial
    apk = args.apk.expanduser().resolve() if args.apk else None
    if apk and (not apk.is_file() or apk.suffix.lower() != ".apk"):
        raise RuntimeError(f"APK не найден: {apk}")
    devices = adb("devices", "-l")
    rows = [line.split() for line in devices.splitlines()[1:] if line.strip()]
    # adbd changes USB interfaces when switching transport modes.
    if not args.wifi_only and not any(row[0] == serial for row in rows):
        for _ in range(6):
            time.sleep(0.5)
            devices = adb("devices", "-l")
            rows = [line.split() for line in devices.splitlines()[1:] if line.strip()]
            if any(row[0] == serial for row in rows):
                break
    usb = any(len(row) > 1 and row[0] == serial and row[1] == "device" for row in rows)
    usb_unauthorized = any(len(row) > 1 and row[0] == serial and row[1] == "unauthorized" for row in rows)
    ips = []
    if usb and not args.wifi_only:
        addr = adb("-s", serial, "shell", "ip", "-o", "-4", "addr", "show", "wlan0")
        match = re.search(r"\binet (\d+\.\d+\.\d+\.\d+)/", addr)
        if not match:
            raise RuntimeError("У Quest нет Wi-Fi IPv4. Подключите шлем к роутеру и повторите команду.")
        ips.append(match.group(1))
    else:
        if args.ip:
            ips.append(str(args.ip))
        ips.extend(row[0].rsplit(":", 1)[0] for row in rows
                   if len(row) > 1 and row[1] == "device" and row[0].endswith(":5555"))
        try:
            saved = json.loads(STATE.read_text())
            if saved.get("serial") == serial:
                ips.append(str(ipaddress.IPv4Address(saved["ip"])))
        except (OSError, ValueError, KeyError):
            pass
        services = adb("mdns", "services", required=False)
        for line in services.splitlines():
            if serial in line and "_adb._tcp" in line:
                match = re.search(r"(\d+\.\d+\.\d+\.\d+):5555", line)
                if match:
                    ips.append(match.group(1))
    target = None
    enabled_usb = False
    for ip in dict.fromkeys(ips):
        print(f"Проверяю Quest: {ip}:5555…", flush=True)
        target = connect(ip, serial)
        if target:
            break
    if not target and usb and not args.wifi_only:
        print("Включаю Wi-Fi ADB через USB…", flush=True)
        print(adb("-s", serial, "tcpip", "5555"), flush=True)
        enabled_usb = True
        for _ in range(5):
            time.sleep(1)
            target = connect(ips[0], serial)
            if target:
                break
    if not target:
        if usb_unauthorized and not args.wifi_only:
            raise RuntimeError("В шлеме подтвердите Allow debugging / Always allow from this computer и повторите команду.")
        raise RuntimeError("Wi-Fi ADB недоступен. После перезапуска подключите Quest по USB, "
                           "разбудите шлем и повторите команду без --wifi-only. "
                           "Компьютер и Quest должны иметь доступ друг к другу в сети роутера. "
                           "При смене IP без USB используйте --ip НОВЫЙ_IP.")
    state = {"serial": serial, "ip": target.rsplit(":", 1)[0], "target": target,
             "connected_utc": datetime.now(timezone.utc).isoformat(),
             "enabled_via_usb_this_run": enabled_usb, "apk": str(apk) if apk else None,
             "installation": "not_requested"}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(f"Quest {serial} подключён по Wi-Fi: {target}. USB можно отключить.", flush=True)
    if apk:
        print(f"Обновляю приложение: {apk}", flush=True)
        result = adb("-s", target, "install", "-r", str(apk), timeout=240)
        if not re.search(r"^Success$", result, flags=re.MULTILINE):
            raise RuntimeError(f"Установка не подтверждена: {result}")
        state["installation"] = "success"
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        print("APK установлен через install -r. Приложение можно открыть в шлеме.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
