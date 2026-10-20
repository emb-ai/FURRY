"""Reproduce the four physically checked same-face cycles; simulation only."""
from pathlib import Path
import os
for key in ['ADJUST','BLEND','DX','DZ','FINGER_MODE','FREEZE_ROT','FREEZE_AFTER','GRIP','GX','GY','HAND','HOLD_GAIN','PRELOAD','PROX','RETREAT','SERVO','SUPPORT_AFTER','SUPPORT_METRIC','SUPPORT_X','SUPPORT_Y','SUPPORT_Z','TRACK','NOSLIP','FORCE']:
 os.environ.pop(key,None)
os.environ.update(BRIDGE='bridge-original-snapshot.dylib',RUBIK_RUN='verified-rerun',ROLL='-45',REBASE='1',PIVOT_X='0',RECOVER='3',TOL_MM='8',DIRECTION='1',WARM_CYCLES='5',TURNS='4',SUPPORT_AFTER='0',SUPPORT_METRIC='14',SUPPORT_Z='.008')
exec((Path(__file__).resolve().parent/'turn_series.py').read_text())
