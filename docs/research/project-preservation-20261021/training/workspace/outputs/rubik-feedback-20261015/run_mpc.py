exec((O/'contact_mpc.py').read_text(),globals())
try:accepted,mpcHistory=predictive_return()
finally:
 np.savez_compressed(O/'mpc-five.npz',trace=trace,qpos=qs,qvel=vs,userdata=states,support=support,commands=commands,actions=actions)
 (O/'mpc-five.json').write_text(json.dumps(events,indent=2))
 if 'mpcHistory' in globals():(O/'mpc-five-control.json').write_text(json.dumps(dict(accepted=accepted,history=mpcHistory,metric=posemetric()),indent=2))
