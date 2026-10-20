from common import *
name=sys.argv[1];z=np.load(O/(name+'.npz'));s=Sim('rubik');s.step(n=2500);maxq=maxu=0.;first=None
try:
 for k,c in enumerate(z['commands']):
  s.step(c[:14].reshape(2,7),c[-2:]);err=float(np.max(abs(s.q-z['qpos'][k])));ue=float(np.max(abs(s.state()-z['userdata'][k])));maxq=max(maxq,err);maxu=max(maxu,ue)
  if err>1e-8 and first is None:first=k
 result=dict(run=name,bridge=os.environ.get('BRIDGE','bridge-home.dylib'),frames=len(z['commands']),qpos_max_error=maxq,userdata_max_error=maxu,first_divergent_frame=first,warnings=int(lib.warnings(s.p)))
 (O/(name+'-replay.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
finally:s.close()
