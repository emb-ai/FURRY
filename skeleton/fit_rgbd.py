"""Fit a replay result with an explicit, disjoint calibration interval."""
import argparse
import json
from pathlib import Path
from human_fit import HumanLegFit,NAMES

INDICES=(11,12,13,14,15,16)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('input');p.add_argument('--output',required=True)
    p.add_argument('--calibration-start',type=float,required=True)
    p.add_argument('--calibration-end',type=float,required=True)
    p.add_argument('--profile');p.add_argument('--use-prior',action='store_true')
    a=p.parse_args()
    if not 0<=a.calibration_start<a.calibration_end:p.error('Invalid calibration interval')
    rows=[json.loads(line) for line in Path(a.input).read_text().splitlines()]
    if not rows:p.error('Empty recording')
    start=rows[0]['t']
    def packet(row):
        return {'t':row['t'],'joints':{n:row.get('depth_points',{}).get(str(i),
                {'p':[0,0,0],'conf':0,'src':'missing'}) for n,i in zip(NAMES,INDICES)}}
    calibration=[packet(r) for r in rows if a.calibration_start*1000<=r['t']-start<a.calibration_end*1000]
    fit=HumanLegFit(json.loads(Path(a.profile).read_text())['lengths_m']) if a.profile else HumanLegFit.calibrate(calibration)
    count=accepted=predicted=0
    with Path(a.output).open('w') as output:
        for row in rows:
            if row['t']-start<a.calibration_end*1000:continue
            prior={n:row['monocular_prior_points'][str(i)] for n,i in zip(NAMES,INDICES)
                   if str(i) in row.get('monocular_prior_points',{})} if a.use_prior else None
            result=fit.fit(packet(row),prior);count+=1;accepted+=result['accepted']
            predicted+=result['accepted'] and 'predicted' in result['sources'].values()
            output.write(json.dumps({'t':row['t'],'file':row['file'],'raw':packet(row),'fit':result})+'\n')
    print(json.dumps({'frames':count,'accepted':accepted,'with_predicted_points':predicted,
                      'lengths_m':fit.lengths.tolist(),'warning':'No ground-truth pose accuracy measured.'},indent=2))

if __name__=='__main__':main()
