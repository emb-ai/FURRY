from pathlib import Path
p=Path(__file__).resolve().parent
exec((p/'geometry_probe.py').read_text().split('results=[]')[0])
from scipy.optimize import minimize
wp0=wp.copy();evaluations=0

def gap(x):
 global wp,evaluations
 wp=wp0+r.apply(x[3:]);evaluations+=1;return quality(x[:3])[0]
initial=np.array([-.13,.70,-.18,-.01,-.005,.005]);opt=minimize(lambda x:float(np.dot(x[3:],x[3:])+.000001*np.dot(x[:3],x[:3])),initial,method='SLSQP',bounds=[(-1.,1.),(-1.,.70),(-1.392,0),(-.04,.01),(-.04,.01),(-.03,.03)],constraints=[{'type':'ineq','fun':lambda x:gap(x)-.003}],options={'maxiter':80,'ftol':1e-9,'eps':1e-4});result=dict(success=bool(opt.success),message=str(opt.message),thumb_angles=opt.x[:3].tolist(),cube_local_wrist_offset_m=opt.x[3:].tolist(),predicted_clearance_m=gap(opt.x),evaluations=evaluations,objective=float(opt.fun));(p/'approach-plan-joints.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
