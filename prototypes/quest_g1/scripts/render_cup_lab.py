"""Render the Quest mesh stream and exact GLSL lighting on desktop OpenGL.

Only the GLSL platform preamble changes. This verifies native asset/shader
appearance, not stereo performance or passthrough composition on Quest.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

import glfw
import mujoco
import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_sim.controller import Controller
from g1_sim.scene import make_model
from g1_sim.visuals import mesh_stream
from scripts.camera_fade_native import CameraFade


def shader_sources():
    source = (ROOT/"android/native/cup_lab_shaders.h").read_text()
    return [re.sub(r"precision highp float;", "", s.replace("#version 320 es", "#version 330 core"))
            for s in re.findall(r'R"\((.*?)\)"', source, re.S)]


def primitive(kind):
    vertices = []
    def tri(a, b, c):
        n = np.cross(np.subtract(b,a),np.subtract(c,a)).astype(float)
        n /= max(np.linalg.norm(n), 1e-12)
        vertices.extend([(*p,*n) for p in (a,b,c)])
    if kind in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_PLANE):
        points = [(1 if i&1 else -1,1 if i&2 else -1,1 if i&4 else -1) for i in range(8)]
        for a,b,c,d in [(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]:
            tri(points[a],points[b],points[c]);tri(points[a],points[c],points[d])
    else:
        sphere = kind in (mujoco.mjtGeom.mjGEOM_SPHERE,mujoco.mjtGeom.mjGEOM_ELLIPSOID)
        rings = 8 if sphere else 1
        def point(i,j):
            a = 2*np.pi*i/16
            z = -np.cos(np.pi*j/rings) if sphere else j*2-1
            r = np.sin(np.pi*j/rings) if sphere else 1
            return (r*np.cos(a),r*np.sin(a),z)
        for j in range(rings):
            for i in range(16):
                a,b,c,d = point(i,j),point(i+1,j),point(i+1,j+1),point(i,j+1)
                tri(a,b,c);tri(a,c,d)
        if not sphere:
            for i in range(16):
                tri((0,0,-1),point(i+1,0),point(i,0));tri((0,0,1),point(i,1),point(i+1,1))
    return np.asarray(vertices,dtype=np.float32)


def camera(eye, center, width, height, fov):
    eye, center = np.array(eye,dtype=float),np.array(center,dtype=float)
    f = center-eye;f /= np.linalg.norm(f)
    s = np.cross(f,[0,0,1]);s /= np.linalg.norm(s)
    u = np.cross(s,f)
    view = np.eye(4);view[:3,:3] = [s,u,-f];view[:3,3] = -view[:3,:3]@eye
    scale = 1/np.tan(np.deg2rad(fov)/2)
    projection = np.zeros((4,4));near,far = .02,60
    projection[0,0]=scale*height/width;projection[1,1]=scale
    projection[2,2]=-(far+near)/(far-near);projection[2,3]=-2*far*near/(far-near);projection[3,2]=-1
    return projection@view


def render(model, data, output, eye, center, size=(1600,1000), fov=58, passthrough=False, camera_fade=False, hide_head=False, clear_color=(.18,.23,.25,1)):
    width,height=size
    if not glfw.init():
        raise RuntimeError("Cannot initialize desktop OpenGL")
    glfw.window_hint(glfw.VISIBLE,glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3);glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT,glfw.TRUE)
    window=glfw.create_window(width,height,"FURRY cup lab QA",None,None)
    if not window:
        glfw.terminate();raise RuntimeError("Cannot create OpenGL core context")
    glfw.make_context_current(window)
    initial_vao=GL.glGenVertexArrays(1);GL.glBindVertexArray(initial_vao)
    program=compileProgram(*[compileShader(s,t) for s,t in zip(shader_sources(),[GL.GL_VERTEX_SHADER,GL.GL_FRAGMENT_SHADER])])
    framebuffer=GL.glGenFramebuffers(1);GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,framebuffer)
    texture=GL.glGenTextures(1);GL.glBindTexture(GL.GL_TEXTURE_2D,texture)
    GL.glTexImage2D(GL.GL_TEXTURE_2D,0,GL.GL_RGBA8,width,height,0,GL.GL_RGBA,GL.GL_UNSIGNED_BYTE,None)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,GL.GL_COLOR_ATTACHMENT0,GL.GL_TEXTURE_2D,texture,0)
    depth=GL.glGenRenderbuffers(1);GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,depth)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER,GL.GL_DEPTH_COMPONENT24,width,height)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,GL.GL_DEPTH_ATTACHMENT,GL.GL_RENDERBUFFER,depth)
    assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)==GL.GL_FRAMEBUFFER_COMPLETE
    GL.glViewport(0,0,width,height);GL.glEnable(GL.GL_DEPTH_TEST);GL.glDisable(GL.GL_CULL_FACE)
    GL.glClearColor(*clear_color);GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(program)
    locations={name:GL.glGetUniformLocation(program,name) for name in ["vp","model","sceneModel","normalMatrix","color","eyePosition","keyDirection","upDirection","materialParams","cupShade","cupGoal","robotPosition","surfaceKind","labStyle"]}
    def mat(name,m):
        GL.glUniformMatrix4fv(locations[name],1,GL.GL_FALSE,np.asarray(m,dtype=np.float32).T.copy())
    mat("vp",camera(eye,center,width,height,fov))
    GL.glUniform3f(locations["eyePosition"],*eye);GL.glUniform3f(locations["keyDirection"],-2,-3,4)
    GL.glUniform3f(locations["upDirection"],0,0,1)
    goal_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_NUMERIC,"cup_lab_goal")
    lab_style=goal_id>=0
    goal=model.numeric_data[model.numeric_adr[goal_id]:model.numeric_adr[goal_id]+4] if lab_style else np.zeros(4)
    cup=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"cup")
    prop=cup if cup>=0 else mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"rubik_core")
    p=data.xpos[prop].copy() if prop>=0 else np.zeros(3)
    placed=cup>=0 and np.linalg.norm(p[:2]-goal[:2])<goal[3] and abs(p[2]-goal[2])<.012 and data.xmat[prop,8]>.966
    if cup<0 and prop>=0:
        p[2]=min(data.geom_xpos[model.geom(f"rubik_block_{i}").id,2]-np.abs(data.geom_xmat[model.geom(f"rubik_block_{i}").id].reshape(3,3)[2])@model.geom_size[model.geom(f"rubik_block_{i}").id] for i in range(8))
    GL.glUniform4f(locations["cupShade"],*p,float(placed));GL.glUniform4f(locations["cupGoal"],*goal)
    GL.glUniform2f(locations["robotPosition"],*(data.qpos[:2] if model.nq>=2 else np.zeros(2)));GL.glUniform1i(locations["labStyle"],int(lab_style))
    scene=mujoco.MjvScene(model,maxgeom=2048);option=mujoco.MjvOption();option.geomgroup[3]=0
    mujoco.mjv_updateScene(model,data,option,None,mujoco.MjvCamera(),mujoco.mjtCatBit.mjCAT_ALL,scene)
    buffers=[];triangles=0;items=[];segment_alpha={};faded_bodies=set()
    fade=CameraFade() if camera_fade else None
    root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"pelvis")
    def robot_body(body):
        while body>0:
            if body==root:return True
            body=int(model.body_parentid[body])
        return False
    for g in scene.geoms[:scene.ngeom]:
        if g.rgba[3]<.01:continue
        body=int(model.geom_bodyid[g.objid]) if g.objid>=0 else -1
        if hide_head and body>=0 and model.body(body).name=="head_link":continue
        if passthrough and (g.type==mujoco.mjtGeom.mjGEOM_PLANE or model.body(model.geom_bodyid[g.objid]).name in ("cup_lab_room","cup_lab_storage")):continue
        vertices=mesh_stream(model,g.dataid//2).reshape(-1,6) if g.type==mujoco.mjtGeom.mjGEOM_MESH else primitive(g.type)
        scale=np.ones(3) if g.type==mujoco.mjtGeom.mjGEOM_MESH else np.array([15,15,.005]) if g.type==mujoco.mjtGeom.mjGEOM_PLANE else g.size
        transform=np.eye(4);transform[:3,:3]=g.mat.reshape(3,3)@np.diag(scale);transform[:3,3]=g.pos
        if g.type==mujoco.mjtGeom.mjGEOM_PLANE:transform[2,3]-=.005
        alpha=float(g.rgba[3])
        if fade and body>=0 and robot_body(body):
            opacity=fade.alpha(transform,vertices,eye);alpha*=opacity
            if opacity<.999:
                mesh_name=model.mesh(g.dataid//2).name if g.type==mujoco.mjtGeom.mjGEOM_MESH else None
                key=model.geom(g.objid).name or mesh_name or model.body(body).name
                segment_alpha[key]=opacity;faded_bodies.add(model.body(body).name)
        items.append([g,vertices,transform,body,alpha])
    items.sort(key=lambda item:(item[4]<.999,-np.sum((item[0].pos-np.asarray(eye))**2) if item[4]<.999 else 0))
    GL.glDisable(GL.GL_BLEND);GL.glDepthMask(GL.GL_TRUE);blending=False
    for g,vertices,transform,body,alpha in items:
        if alpha<.01:continue
        if not blending and alpha<.999:
            GL.glEnable(GL.GL_BLEND);GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA,GL.GL_ONE_MINUS_SRC_ALPHA,GL.GL_ONE,GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDepthMask(GL.GL_FALSE);blending=True
        mat("model",transform);mat("sceneModel",transform)
        normal=np.linalg.inv(transform[:3,:3]).T
        GL.glUniformMatrix3fv(locations["normalMatrix"],1,GL.GL_FALSE,normal.astype(np.float32).T.copy())
        GL.glUniform4f(locations["color"],*g.rgba[:3],alpha)
        GL.glUniform3f(locations["materialParams"],g.specular,g.shininess,g.emission)
        mid=model.geom_matid[g.objid] if g.objid>=0 else -1
        material=model.material(mid).name if mid>=0 else ""
        surface={"lab_floor":1,"lab_wood":2,"lab_ceramic":3,"lab_goal":4,"lab_window":5}.get(material,0)
        GL.glUniform1i(locations["surfaceKind"],surface)
        vao=GL.glGenVertexArrays(1);vbo=GL.glGenBuffers(1);buffers.append((vao,vbo))
        GL.glBindVertexArray(vao);GL.glBindBuffer(GL.GL_ARRAY_BUFFER,vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER,vertices.nbytes,vertices,GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0);GL.glVertexAttribPointer(0,3,GL.GL_FLOAT,GL.GL_FALSE,24,None)
        from ctypes import c_void_p
        GL.glEnableVertexAttribArray(1);GL.glVertexAttribPointer(1,3,GL.GL_FLOAT,GL.GL_FALSE,24,c_void_p(12))
        GL.glDrawArrays(GL.GL_TRIANGLES,0,len(vertices));triangles+=len(vertices)//3
    GL.glDepthMask(GL.GL_TRUE);GL.glDisable(GL.GL_BLEND)
    pixels=GL.glReadPixels(0,0,width,height,GL.GL_RGBA,GL.GL_UNSIGNED_BYTE)
    rgba_center=np.frombuffer(pixels,dtype=np.uint8).reshape(height,width,4)[height//2,width//2].tolist()
    Image.frombytes("RGBA",(width,height),pixels).transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB").save(output)
    depth_center=float(np.frombuffer(GL.glReadPixels(width//2,height//2,1,1,GL.GL_DEPTH_COMPONENT,GL.GL_FLOAT),dtype=np.float32)[0])
    error=GL.glGetError();assert error==GL.GL_NO_ERROR,error
    renderer=GL.glGetString(GL.GL_RENDERER).decode()
    for vao,vbo in buffers:GL.glDeleteVertexArrays(1,[vao]);GL.glDeleteBuffers(1,[vbo])
    GL.glDeleteProgram(program);GL.glDeleteTextures([texture]);GL.glDeleteRenderbuffers(1,[depth]);GL.glDeleteFramebuffers(1,[framebuffer])
    GL.glDeleteVertexArrays(1,[initial_vao]);glfw.destroy_window(window);glfw.terminate()
    return {"triangles":triangles,"renderer":renderer,"native_shader":True,"headset_capture":False,"segment_alpha":segment_alpha,"faded_geom_bodies":sorted(faded_bodies),"camera_fade":camera_fade,"depth_center":depth_center,"rgba_center":rgba_center}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("output",type=Path);parser.add_argument("--before",type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    m,_=make_model("cup");d=mujoco.MjData(m);Controller(m).initialize(d)
    views={"cup-lab-overview":((1.85,-2.15,1.85),(.65,.15,.95),58),
           "cup-lab-workbench":((.05,-.72,1.14),(.62,-.05,.77),52),
           "cup-lab-room":((-1.9,-1.8,1.8),(.9,.25,1.1),67)}
    results={}
    for name,(eye,center,fov) in views.items():results[name]=render(m,d,args.output/(name+".png"),eye,center,fov=fov)
    results["cup-lab-mr"]=render(m,d,args.output/"cup-lab-mr.png",views["cup-lab-overview"][0],views["cup-lab-overview"][1],passthrough=True)
    if args.before:
        spec=importlib.util.spec_from_file_location("g1_sim.before_scene",args.before);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        old,_=module.make_model("cup");old_data=mujoco.MjData(old);Controller(old).initialize(old_data)
        results["cup-lab-before"]=render(old,old_data,args.output/"cup-lab-before.png",views["cup-lab-overview"][0],views["cup-lab-overview"][1])
    (args.output/"render-verification.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":main()
