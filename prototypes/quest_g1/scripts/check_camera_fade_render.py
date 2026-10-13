"""Native-shader desktop QA, including actual blending/depth behaviour."""
import argparse
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1_sim.scene import make_model
from g1_sim.controller import Controller
from g1_sim.visuals import mesh_stream
from scripts.render_cup_lab import render


def main(out):
    out.mkdir(parents=True,exist_ok=True);m,_=make_model('cup');d=mujoco.MjData(m);Controller(m).initialize(d)
    original=d.qpos.copy();results={}
    head=next(g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name=='head_link')
    head_vertices=mesh_stream(m,m.geom_dataid[head]).reshape(-1,6)[:,:3]
    head_world=head_vertices@d.geom_xmat[head].reshape(3,3).T+d.geom_xpos[head]
    above=d.geom_xpos[head].copy();above[0]-=.04;above[2]=head_world[:,2].max()+.04
    views={
        'head':(d.geom_xpos[head]+[-.06,0,0],np.array([.65,0,1.1])),
        'head-above':(above,np.array([.65,0,1.1])),
        'torso':(d.xpos[m.body('torso_link').id]+[.03,0,.10],np.array([.62,0,.83])),
        'wrist':(d.xpos[m.body('left_wrist_yaw_link').id]+[-.05,0,0],np.array([.58,0,.84])),
        'far':(np.array([1.8,-2.1,1.8]),np.array([.65,0,.95])),
    }
    for name,(eye,target) in views.items():
        before=render(m,d,out/(name+'-before.png'),eye,target,size=(960,600),fov=75)
        after=render(m,d,out/(name+'-after.png'),eye,target,size=(960,600),fov=75,camera_fade=True)
        assert np.array_equal(original,d.qpos),'Renderer altered physics'
        assert 'cup' not in after['faded_geom_bodies'] and 'table' not in after['faded_geom_bodies'],'Prop faded'
        if name=='head':assert after['segment_alpha'].get('head_link',1)==0
        if name=='head-above':
            assert after['segment_alpha'].get('head_link',1)==0
            assert after['segment_alpha'].get('torso_link',1)>.9,'Head fade hid the shared-body torso mesh'
        if name=='torso':assert after['segment_alpha'].get('torso_link',1)==0
        if name=='wrist':assert after['segment_alpha'].get('left_wrist_yaw_link',1)==0
        if name=='far':assert not after['segment_alpha']
        a=np.asarray(Image.open(out/(name+'-before.png')),dtype=float);b=np.asarray(Image.open(out/(name+'-after.png')),dtype=float)
        difference=float(np.mean(np.abs(a-b)))
        if name=='far':assert difference==0,'Far appearance changed'
        else:assert difference>1,'Near segment still occludes the image'
        results[name]=dict(before=before,after=after,mean_pixel_difference=difference)
    # A red robot segment in front of an opaque green prop. Test genuine
    # framebuffer blending, not just a computed alpha uniform.
    xml='<mujoco><worldbody><body name="pelvis"><body name="head_link"><geom type="box" pos=".2 0 0" size=".05 .12 .12" rgba="1 0 0 1"/></body></body><body name="prop"><geom type="box" pos=".45 0 0" size=".10 .12 .12" rgba="0 1 0 1"/></body></worldbody></mujoco>'
    fixture=mujoco.MjModel.from_xml_string(xml);fd=mujoco.MjData(fixture);mujoco.mj_forward(fixture,fd)
    eye=[0,0,0];target=[1,0,0]
    render(fixture,fd,out/'blend-before.png',eye,target,size=(256,256),camera_fade=False)
    blended=render(fixture,fd,out/'blend-after.png',eye,target,size=(256,256),camera_fade=True)
    old=np.asarray(Image.open(out/'blend-before.png'));new=np.asarray(Image.open(out/'blend-after.png'))
    assert old[128,128,1]==0 and new[128,128,1]>15,'Opaque foreground depth hid the green prop'
    assert abs(blended['segment_alpha']['head_link']-.5)<1e-5
    fixture.geom_rgba[0,3]=0
    reference=render(fixture,fd,out/'blend-reference.png',eye,target,size=(256,256),camera_fade=False)
    assert abs(blended['depth_center']-reference['depth_center'])<1e-6,'Transparent segment wrote depth'
    results['depth_blending']=dict(before_pixel=old[128,128].tolist(),after_pixel=new[128,128].tolist(),head_alpha=.5,depth_center=blended['depth_center'],opaque_reference_depth=reference['depth_center'])
    fixture.geom_rgba[0,3]=1;fixture.geom_rgba[1,3]=0
    alpha_frame=render(fixture,fd,out/'premultiplied-frame.png',eye,target,size=(256,256),camera_fade=True,clear_color=(0,0,0,0))
    rgba=alpha_frame['rgba_center'];assert 185<=rgba[3]<=195 and 0<rgba[0]<=rgba[3] and rgba[1]==0
    results['premultiplied_frame']=dict(rgba_center=rgba,openxr_unpremultiplied_flag=False)
    fixture.geom_rgba[0,3]=0;fixture.geom_rgba[1,3]=1;fixture.geom_pos[1]=[.2,0,0];fixture.geom_size[1]=[.05,.12,.12];mujoco.mj_forward(fixture,fd)
    close_before=render(fixture,fd,out/'close-prop-before.png',eye,target,size=(256,256))
    close_after=render(fixture,fd,out/'close-prop-after.png',eye,target,size=(256,256),camera_fade=True)
    assert close_before['rgba_center']==close_after['rgba_center'] and not close_after['segment_alpha'],'Near prop was faded'
    results['near_prop']=dict(rgba_center=close_after['rgba_center'],unchanged=True)
    (out/'render-checks.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();main(args.output)
