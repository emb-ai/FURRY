"""Summary table of the study variants from m_<name>.json, r_<name>.json and q_<name>.log.

usage (in the output folder): compare.py name ...
Held-out = stages 2 and 9, the walking stages excluded from the quest-111260 fine-tune.
"""
import json, re, sys

HELD = (2, 9)


def held(m):
    st = [v for k, v in m['stages'].items() if int(k) in HELD]
    return sum(v['trips_per_s'] * v['seconds'] for v in st) / sum(v['seconds'] for v in st)


cols = ['variant', 'trips/s', 'held-out 2+9', 'braking/s', 'touchdowns/s', 'robot swing peak cm',
        'swings <1 cm', 'swing on floor', 'knee lag ms', 'falls', 'drift m', 'path ratio', 'heading err deg']
print('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols))
for v in sys.argv[1:]:
    m, r = json.load(open('m_%s.json' % v)), json.load(open('r_%s.json' % v))
    falls = re.findall(r'falls=(\d+)', open('q_%s.log' % v).read())[-1]
    row = [v, m['trips_per_s'], round(held(m), 3), m['braking_trips_per_s'], m['touchdowns_per_s'],
           m['phys_peak_median_cm'], m['swings_phys_below_1cm'], m['swing_floor_contact_share'],
           m['knee_lag_ms_median'], falls, r['drift_m'], r['path_ratio'], r['heading_err_deg']]
    print('| ' + ' | '.join(str(x) for x in row) + ' |')
