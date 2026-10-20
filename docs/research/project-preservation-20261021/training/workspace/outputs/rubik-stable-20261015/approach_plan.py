from pathlib import Path
p=Path(__file__).resolve().parent
exec((p/'geometry_probe.py').read_text().split('results=[]')[0])
from scipy.optimize import minimize
wp0=wp.copy();evaluations=0

def gap(x):
 global wp,evaluations
 wp=wp0+r.apply(x[2:]);evaluations+=1;return quality([0,x[0],x[1]])[0]
initial=np.array([0.,0.,-.04,-.015,0.]);opt=minimize(lambda x:float(np.dot(x[2:],x[2:])+.00001*np.dot(x[:2],x[:2])),initial,method='SLSQP',bounds=[(-.8,0),(-1.392,0),(-.06,.01),(-.04,.01),(-.04,.04)],constraints=[{'type':'ineq','fun':lambda x:gap(x)-.003}],options={'maxiter':80,'ftol':1e-9,'eps':1e-4});result=dict(success=bool(opt.success),message=str(opt.message),thumb_angles=[0,float(opt.x[0]),float(opt.x[1])],cube_local_wrist_offset_m=opt.x[2:].tolist(),predicted_clearance_m=gap(opt.x),evaluations=evaluations,objective=float(opt.fun));(p/'approach-plan.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
