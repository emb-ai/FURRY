import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Camera skeleton + Quest trajectory logger')
    commands = parser.add_subparsers(dest='command', required=True)
    record = commands.add_parser('record', help='RealSense raw + YOLO RGB-D + AprilTags + Quest WebSocket')
    record.add_argument('--output', required=True, help='New session directory (must not exist)')
    record.add_argument('--model', required=True, help='Existing local YOLO pose model or OpenVINO directory')
    record.add_argument('--tag-layout', help='Measured marker size/layout JSON; omitted means 2D tags only')
    record.add_argument('--serial')
    record.add_argument('--camera-id', default='front')
    record.add_argument('--host', default='0.0.0.0')
    record.add_argument('--port', type=int, default=8765)
    record.add_argument('--width', type=int, default=1280)
    record.add_argument('--height', type=int, default=720)
    record.add_argument('--fps', type=int, default=30)
    record.add_argument('--seconds', type=float, default=0, help='0 = until Ctrl+C')
    for name in ('validate', 'qc', 'verify-bag'):
        commands.add_parser(name).add_argument('session')
    imp = commands.add_parser('import-quest')
    imp.add_argument('session')
    imp.add_argument('episode')
    export = commands.add_parser('export')
    export.add_argument('session')
    export.add_argument('--output', required=True)
    export.add_argument('--max-gap-ms', type=float, default=50)
    patch = commands.add_parser('quest-patch')
    patch.add_argument('source', help='Path to camera_stream.cpp')
    patch.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'record':
        if args.seconds < 0 or min(args.width, args.height, args.fps) <= 0:
            parser.error('Invalid capture dimensions/rate/duration')
        from .capture import record
        record(args)
    elif args.command == 'quest-patch':
        from .quest_patch import generate
        print(generate(args.source, args.output))
    else:
        from .offline import validate, export, import_episode, verify_bag
        from .storage import save_json
        if args.command == 'import-quest':
            print(import_episode(args.session, args.episode))
            return
        if args.command == 'export':
            if args.max_gap_ms <= 0:
                parser.error('--max-gap-ms must be positive')
            result = export(args.session, args.output, args.max_gap_ms)
        elif args.command == 'verify-bag':
            result = verify_bag(args.session)
        else:
            result = validate(args.session)
            if args.command == 'qc':
                report = Path(args.session)/'reports'
                report.mkdir(exist_ok=True)
                save_json(report/'qc.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get('valid') is False:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
