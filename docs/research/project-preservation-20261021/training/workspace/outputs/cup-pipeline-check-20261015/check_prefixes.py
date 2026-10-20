from pathlib import Path
s=Path(__file__).with_name('check_replay.py').read_text().replace("manifest['splits']['valid']+manifest['splits']['train'][:1]","manifest['splits']['valid'][2:3]").replace('for warm in [0,.2,.5]:','for warm in [.8,1.0]:').replace("'replay-check.json'","'prefix-check.json'")
exec(compile(s,str(Path(__file__).with_name('check_replay.py')),'exec'))
