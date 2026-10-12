"""Learning curves, phase-separated; critic-only surrogate placeholders excluded."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser()
for n in ('control','faults','history-control','history-faults'):p.add_argument('--'+n,type=Path)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
fig,axes=plt.subplots(3,3,figsize=(16,11),constrained_layout=True)
colors={'control':'#2479b9','faults':'#e78020'}
summary={}
for name in colors:
 files=[getattr(a,'history_'+name),getattr(a,name)]
 rows=[]
 for f in files:
  if f and f.exists():
   for line in f.read_text().splitlines():
    try:r=json.loads(line)
    except json.JSONDecodeError:continue
    if 'value_loss' in r:rows.append(r)
 # The pilot and continuation use cumulative optimizer/actor counters.
 rows=list({(r['phase'],r['iteration']):r for r in rows}.values())
 if not rows:continue
 critic=[r for r in rows if r['phase']=='critic'];actor=[r for r in rows if r['phase']=='actor']
 specs=[(0,0,critic,'value_loss','Critic warmup: value loss'),(0,1,critic,'pre_update_critic_ev','Critic warmup: explained variance'),(0,2,actor,'value_loss','Actor phase: value loss'),(1,0,actor,'surrogate_loss','Actor phase: PPO surrogate loss'),(1,1,actor,'reward_rate','Training reward / second'),(1,2,actor,'window_distance_error_m','Window displacement error (m)'),(2,0,actor,'lr','Adaptive learning rate'),(2,1,actor,'std_mean','Mean exploration std'),(2,2,actor,'pre_update_critic_ev','Actor phase: critic explained variance')]
 for row,col,data,key,title in specs:
  ax=axes[row,col];ax.set_title(title,fontsize=11);ax.grid(alpha=.2)
  iscritic=row==0 and col<2
  ax.set_xlabel('Critic updates' if iscritic else 'Actor updates (cumulative)')
  if not data:continue
  x=np.array([r['iteration'] if iscritic else r['actor_updates']+1 for r in data]);y=np.array([r[key] for r in data])
  ax.plot(x,y,color=colors[name],alpha=.2,lw=.7)
  # Trailing average has no future-data leakage; only full windows shown.
  window=min(50,len(y));smooth=np.convolve(y,np.ones(window)/window,'valid')
  ax.plot(x[window-1:],smooth,color=colors[name],lw=1.8,label=name+' (mean %d)'%window)
  if getattr(a,'history_'+name) and not iscritic:
   ax.axvline(500,color='gray',ls=':',lw=1)
   ax.axvline(1000,color='gray',ls='--',lw=1)
 summary[name]={'critic_updates':len(critic),'actor_updates':max([r['actor_updates']+1 for r in actor],default=0),'last_100_actor_means':{k:float(np.mean([r[k] for r in actor[-100:]])) for k in ['value_loss','surrogate_loss','reward_rate','window_distance_error_m','lr','std_mean','pre_update_critic_ev']} if actor else {}}
for ax in axes.flat:
 if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
axes[0,0].set_yscale('log');axes[2,0].set_yscale('log')
fig.suptitle('TWIST2 continuation to 5000 actor updates | control vs actor leg delay/freeze\nSince update 1000: 1 H100 per condition; dotted 500 = V100 continuation, dashed 1000 = single-GPU restart\nRaw values + trailing mean; training losses do not measure live trips or task success',fontsize=13)
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160);plt.close(fig)
a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
