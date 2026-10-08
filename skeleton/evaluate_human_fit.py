"""Offline causal leg-fit experiment. Inputs remain private; outputs are not GT."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from human_fit import HumanLegFit, NAMES, EDGES, observed


def packets(path):
    result=[]
    for line in Path(path).read_text().splitlines():
        row=json.loads(line)
        if row.get('direction','downlink')!='downlink':continue
        p=row.get('payload',row)
        if 'joints' in p:result.append(p)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('calibration');ap.add_argument('evaluation');ap.add_argument('output')
    ap.add_argument('--profile',help='JSON with lengths_m: hip width, left/right thigh, left/right shin')
    args=ap.parse_args()
    if Path(args.calibration).resolve()==Path(args.evaluation).resolve():
        ap.error('Use a separate calibration interval, not the evaluation recording')
    fit=(HumanLegFit(json.loads(Path(args.profile).read_text())['lengths_m']) if args.profile
         else HumanLegFit.calibrate(packets(args.calibration)))
    source=packets(args.evaluation);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    ms=[];adjust=[];before=[];after=[];reasons={}
    with (out/'fitted.jsonl').open('w') as f:
        for packet in source:
            begin=time.perf_counter();result=fit.fit(packet);ms.append((time.perf_counter()-begin)*1000)
            reasons[result['reason']]=reasons.get(result['reason'],0)+1
            if result['accepted']:
                adjust.append(result['max_adjustment_m'])
                x=np.array([packet['joints'][n]['p'] for n in NAMES])
                y=np.array([result['points'][n] for n in NAMES])
                before.append([np.linalg.norm(x[a]-x[b]) for a,b in EDGES])
                after.append([np.linalg.norm(y[a]-y[b]) for a,b in EDGES])
            f.write(json.dumps({'t':packet['t'],'raw':packet,'fit':result})+'\n')
    q=lambda v:np.quantile(v,[.5,.95,1]).tolist() if len(v) else []
    summary={'frames':len(source),'reasons':reasons,'lengths_m':fit.lengths.tolist(),
             'solve_ms_p50_p95_max':q(ms),'max_adjustment_m_p50_p95_max':q(adjust),
             'raw_bone_std_m_on_accepted':np.std(before,axis=0).tolist() if before else [],
             'fitted_bone_std_m_on_accepted':np.std(after,axis=0).tolist() if after else [],
             'warning':'Constant fitted lengths are enforced, NOT evidence of position accuracy; no robot replay or raw RGB-D comparison.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
