"""Fetch pinned public dependencies into vendor/; never overwrite an existing checkout."""
from pathlib import Path
import subprocess
import urllib.request
import zipfile
import io

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor'
VENDOR.mkdir(exist_ok=True)
for name, url, rev in [
    ('TWIST2', 'https://github.com/amazon-far/TWIST2.git', 'b06178f19a22f2138cbd31f60c6d494bc263f67d'),
    ('mujoco', 'https://github.com/google-deepmind/mujoco.git', 'f1d45bd5422c74beddfb0d1deb590a02583d21de'),
    ('OpenXR-SDK-Source', 'https://github.com/KhronosGroup/OpenXR-SDK-Source.git', '47c4761d05b27f884247480ac3fb8a6657907325'),
]:
    dest = VENDOR / name
    if not dest.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', url, str(dest)], check=True)
        subprocess.run(['git', '-C', str(dest), 'checkout', rev], check=True)
    head = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
    if head != rev:
        raise SystemExit(f'{name}: expected {rev}, found {head}; preserve checkout and resolve manually')
for name, url in [
    ('onnxruntime-android', 'https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.23.2/onnxruntime-android-1.23.2.aar'),
    ('openxr-android', 'https://repo.maven.apache.org/maven2/org/khronos/openxr/openxr_loader_for_android/1.1.49/openxr_loader_for_android-1.1.49.aar'),
]:
    dest = VENDOR / name
    if dest.exists():
        print(f'Preserving existing {dest}; verify its version before building')
        continue
    data = urllib.request.urlopen(url).read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.infolist():
            if not (dest / entry.filename).resolve().is_relative_to(dest.resolve()):
                raise ValueError('Unsafe archive path')
        archive.extractall(dest)
for name in ['LICENSE', 'ThirdPartyNotices.txt']:
    target = VENDOR / 'onnxruntime-android' / name
    if not target.exists():
        target.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/{name}').read())
print('Sources ready. Install Python dependencies and the Android/JDK toolchain described in android/README.md.')
