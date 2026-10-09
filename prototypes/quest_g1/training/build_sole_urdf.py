"""Copy main's eight MuJoCo sole spheres into the source Isaac Gym URDF.

All other URDF elements, including explicit inertials, must remain identical.
Keep a meshes symlink beside the output pointing to the source G1 meshes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def build(source, scene, output):
    tree, reference = ET.parse(source), ET.parse(scene)
    changes = []
    for side in ('left', 'right'):
        name = side+'_ankle_roll_link'
        body = tree.find(".//link[@name='%s']" % name)
        main_body = reference.find(".//body[@name='%s']" % name)
        old = body.findall('collision')
        spheres = [g for g in main_body.findall('geom') if g.get('type') == 'sphere'
                   and g.get('name', '').startswith(side+'_foot')]
        if len(old) != 2 or len(spheres) != 4:
            raise ValueError('Unexpected source/main sole layout')
        for c in old:
            body.remove(c)
        for sphere in spheres:
            c = ET.SubElement(body, 'collision', name=sphere.attrib['name'])
            ET.SubElement(c, 'origin', xyz=sphere.attrib['pos'], rpy='0 0 0')
            ET.SubElement(ET.SubElement(c, 'geometry'), 'sphere', radius=sphere.attrib['size'])
        changes.append({'link':name, 'old':[ET.tostring(c, encoding='unicode') for c in old],
                        'new':[dict(g.attrib) for g in spheres]})
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding='unicode')
    untouched = ET.parse(source)
    for t in (tree, untouched):
        for side in ('left', 'right'):
            b = t.find(".//link[@name='%s_ankle_roll_link']" % side)
            for c in b.findall('collision'):
                b.remove(c)
    if ET.tostring(tree.getroot()) != ET.tostring(untouched.getroot()):
        raise AssertionError('Non-sole URDF content changed')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {'only_sole_collisions_changed':True, 'source_sha256':sha(source),
              'scene_sha256':sha(scene), 'output_sha256':sha(output), 'changes':changes}
    output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('source', 'scene', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.scene, a.output), indent=2))
