"""Exercise production export JNI and real ZIP streams; mock only Android storage.

Requires JAVA_HOME (or the project's bundled JDK) and a C++17 compiler. This is
not a replacement for testing MediaStore/MTP on a physical Quest.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import zipfile
ROOT=Path(__file__).resolve().parents[1]
jdk=Path(os.environ.get('JAVA_HOME',ROOT/'.tools/jdk/Contents/Home'))
platform='darwin' if sys.platform=='darwin' else 'linux'
fixture=ROOT/'tests/export_jni'
with tempfile.TemporaryDirectory(prefix='g1-export-') as temp:
    work=Path(temp);classes=work/'classes';classes.mkdir();episodes=work/'episodes';episodes.mkdir();shared=work/'shared'
    for name in ('episode-001','episode-002'):
        p=episodes/name;p.mkdir();(p/'input.csv').write_bytes((b'sequence,time\n1,100\n'*10000)+b'end\n');(p/'complete.json').write_text('{"finalized":true}\n')
    subprocess.run([str(jdk/'bin/javac'),'-d',str(classes),*[str(p) for p in fixture.rglob('*.java')]],check=True)
    binary=work/'check-export'
    subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-pthread',
                   '-I'+str(fixture/'stubs'),'-I'+str(ROOT/'android/native'),
                   '-I'+str(jdk/'include'),'-I'+str(jdk/'include'/platform),
                   str(fixture/'check_export.cpp'),'-L'+str(jdk/'lib/server'),'-ljvm',
                   '-Wl,-rpath,'+str(jdk/'lib/server'),'-o',str(binary)],check=True)
    subprocess.run([str(binary),str(classes),str(shared),str(episodes)],check=True)
    assert sorted(p.name for p in shared.iterdir())==['g1-episode-001.zip','g1-episode-002.zip']
    for name in ('episode-001','episode-002'):
        with zipfile.ZipFile(shared/f'g1-{name}.zip') as archive:
            assert archive.testzip() is None
            assert sorted(archive.namelist())==[f'{name}/complete.json',f'{name}/input.csv']
            for field in ('input.csv','complete.json'):
                assert archive.read(f'{name}/{field}')==(episodes/name/field).read_bytes()
print('Export JNI passed: latest selection, multi-buffer ZIP round-trip, deduplication, failure cleanup, original retention.')
