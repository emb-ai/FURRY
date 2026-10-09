"""Select only exact members of pinned TWIST2 train manifest; never re-split them."""
import argparse,collections,hashlib,json,re
from pathlib import Path
REV='b06178f19a22f2138cbd31f60c6d494bc263f67d'
FAMILIES=('OMOMO_g1_GMR','AMASS_g1_GMR8','twist1_to_twist2')

def select(manifest,index,count=1024,seed=20261009):
 rows=re.findall(r'^- file: (.+)\n\s+weight: ([\d.]+)',manifest,re.M)
 if len({x[0] for x in rows})!=len(rows):raise ValueError('Duplicate author manifest paths')
 entries={x['name'].removeprefix('TWIST2_full/'):x for x in index['entries'] if x['name'].startswith('TWIST2_full/') and x['name'].endswith('.pkl')}
 groups={f:[(name,float(w)) for name,w in rows if name.split('/')[0]==f] for f in FAMILIES}
 missing=[n for g in groups.values() for n,w in g if n not in entries]
 if missing:raise ValueError(f'{len(missing)} author train files missing from archive: {missing[:3]}')
 total=sum(len(g) for g in groups.values())
 if not len(FAMILIES)<=count<=total:raise ValueError('Subset count outside available author train pool')
 quotas={f:int(count*len(g)/total) for f,g in groups.items()}
 for f in sorted(FAMILIES,key=lambda f:count*len(groups[f])/total-quotas[f],reverse=True)[:count-sum(quotas.values())]:quotas[f]+=1
 for f in FAMILIES:
  if quotas[f]==0:
   donor=max(FAMILIES,key=lambda g:quotas[g]);quotas[donor]-=1;quotas[f]=1
 selected=[]
 for f,g in groups.items():
  chosen=sorted(g,key=lambda x:hashlib.sha256(f'{seed}:{x[0]}'.encode()).hexdigest())[:quotas[f]]
  mass=sum(w for n,w in g)
  for n,w in chosen:selected.append({'author_path':n,'author_split':'train','local_split':'train','family':f,'author_weight':w,'sampling_weight':mass/len(chosen),'archive':entries[n]})
 pico_mass=sum(float(w) for n,w in rows if n.split('/')[0]=='v1_v2_v3_g1')
 return {'schema':'twist2-author-train-subset-v1','source_revision':REV,'source_manifest_sha256':hashlib.sha256(manifest.encode()).hexdigest(),'source_archive_bytes':index['bytes'],'selection_seed':seed,'selection':'deterministic hash-stratified sample from exact train manifest membership; no policy-performance filtering','family_counts':quotas,'author_non_pico_count':total,'author_non_pico_weight':sum(sum(w for n,w in g) for g in groups.values()),'quest_replacement_weight':pico_mass,'quest_probability_before_curriculum':pico_mass/(pico_mass+sum(sum(w for n,w in g) for g in groups.values())),'excluded_pico_count':sum(n.startswith('v1_v2_v3_g1/') for n,w in rows),'archive_pkls_not_in_author_manifest':len(set(entries)-{n for n,w in rows}),'clips':selected}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=1024);a=p.parse_args()
 if a.count<len(FAMILIES):raise ValueError('Subset must cover all source families')
 result=select(a.manifest.read_text(),json.loads(a.index.read_text()),a.count);a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='clips'},indent=2))
