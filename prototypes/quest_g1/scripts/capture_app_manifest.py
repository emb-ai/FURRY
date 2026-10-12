"""Build a separately signed capture app without replacing the shared runtime."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def generate(source, destination):
    android = 'http://schemas.android.com/apk/res/android'
    ET.register_namespace('android', android)
    tree = ET.parse(source)
    root = tree.getroot()
    if root.get('package') != 'org.g1lab.quest':
        raise ValueError('Unexpected base application identity')
    root.set('package', 'org.g1lab.quest.capture')
    root.find('application').set('{'+android+'}label', 'FURRY Capture')
    tree.write(destination, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path); p.add_argument('destination', type=Path)
    a = p.parse_args(); generate(a.source, a.destination)
