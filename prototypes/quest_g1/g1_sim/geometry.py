"""Hybrid geometry without changing articulation, inertias or control parameters."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ASSETS = Path(__file__).resolve().parents[1] / 'assets/g1'
FLOOR, BODY, HAND, FOOT_BOX, PROP = 1, 2, 4, 8, 16


def apply_hybrid(root, visuals=True):
    description = json.loads((ASSETS/'collision.json').read_text())
    pelvis = root.find(".//body[@name='pelvis']")
    bodies = {b.get('name'): b for b in pelvis.iter('body')}
    for body in bodies.values():
        for geom in list(body.findall('geom')):
            if geom.get('contype', '1') != '0' or geom.get('conaffinity', '1') != '0':
                body.remove(geom)
    asset = root.find('asset')
    if visuals:
        for mesh in asset.findall('mesh'):
            name = mesh.get('name', Path(mesh.get('file')).stem)
            mesh.set('file', str(ASSETS/'visual'/f'{name}.obj'))
    common = dict(group='3', density='0', rgba='.2 .6 .4 .35', condim='3')
    for item in description['body']:
        attrs = item['geom'].copy()
        foot_box = attrs['name'].endswith('foot_box_collision')
        attrs.update(common, contype=str(FOOT_BOX if foot_box else BODY),
                     conaffinity=str(PROP if foot_box else FLOOR | PROP))
        ET.SubElement(bodies[item['body']], 'geom', attrs)
    for hand in description['hands']:
        name = hand['name']
        ET.SubElement(asset, 'mesh', name=name,
                      vertex=' '.join(format(x, '.9g') for v in hand['vertices'] for x in v),
                      face=' '.join(str(x) for f in hand['faces'] for x in f))
        ET.SubElement(bodies[hand['body']], 'geom', common | dict(
            name=name, type='mesh', mesh=name, pos=hand['pos'], quat=hand['quat'],
            contype=str(HAND), conaffinity=str(FLOOR | BODY | HAND | PROP)))
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    for pair in description['body_pairs']:
        ET.SubElement(contact, 'pair', pair)


def configure_props(world):
    # Floor remains category 1; foot boxes contact props, not the floor. MJX
    # sole capsules provide floor contacts without duplicate box contacts.
    for body in world.findall('body'):
        if body.get('name') not in ('table', 'cup', 't_box'):
            continue
        for geom in body.iter('geom'):
            geom.set('contype', str(PROP))
            geom.set('conaffinity', str(FLOOR | BODY | HAND | FOOT_BOX | PROP))
