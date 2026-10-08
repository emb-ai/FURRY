"""Embed two local motion exports into the orbitable human-skeleton viewer."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yolo', type=Path)
    parser.add_argument('rtmw', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    payload = {name: json.loads(getattr(args, name).read_text()) for name in ['yolo', 'rtmw']}
    if abs(payload['yolo']['sourceStart'] - payload['rtmw']['sourceStart']) > .05:
        raise ValueError('Exports must have matching source start timestamps')
    template = Path(__file__).with_name('human_gmr_viewer.html').read_text()
    html = template.replace('__ANIMATION_DATA__', json.dumps(payload, separators=(',', ':')))
    if len(html.encode()) >= 1_000_000:
        raise ValueError('Reduce export size before embedding')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)


if __name__ == '__main__':
    main()
