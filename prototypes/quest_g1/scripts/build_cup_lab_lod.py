"""One-time visual LOD authoring; never run during APK packaging.

Uses installed fast_simplification. Source robot contact geometry is untouched.
Rebuild: .venv/bin/python scripts/build_cup_lab_lod.py
"""
import hashlib
import json
from pathlib import Path

import fast_simplification
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"assets/g1/visual"
DEST=SOURCE/"cup_lab"


def load(path):
    vertices,faces=[],[]
    for line in path.read_text().splitlines():
        fields=line.split()
        if fields and fields[0]=="v":vertices.append(list(map(float,fields[1:4])))
        if fields and fields[0]=="f":faces.append([int(f.split("/")[0])-1 for f in fields[1:4]])
    return np.array(vertices),np.array(faces,dtype=np.int32)


def distance(points, triangles):
    result=[]
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    ab,ac=b-a,c-a
    n=np.cross(ab,ac);n2=np.einsum("ij,ij->i",n,n)
    for start in range(0,len(points),32):
        p=points[start:start+32,None,:]
        ap=p-a
        projected=p-n*np.sum(ap*n,axis=2)[:,:,None]/np.maximum(n2,1e-24)[None,:,None]
        inside=np.ones(projected.shape[:2],dtype=bool)
        edge_dist=np.full(projected.shape[:2],np.inf)
        for x,y in [(a,b),(b,c),(c,a)]:
            edge=y-x
            inside &= np.sum(np.cross(edge,projected-x)*n,axis=2)>=-1e-20
            t=np.clip(np.sum((p-x)*edge,axis=2)/np.maximum(np.sum(edge*edge,axis=1),1e-24),0,1)
            closest=x+t[:,:,None]*edge
            edge_dist=np.minimum(edge_dist,np.sum((p-closest)**2,axis=2))
        plane=np.sum((p-projected)**2,axis=2)
        result.extend(np.sqrt(np.min(np.where(inside,plane,edge_dist),axis=1)))
    return np.array(result)


def write(path, vertices, faces):
    triangles=vertices[faces]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-16)
    incident=[[] for _ in vertices]
    for i,face in enumerate(faces):
        for v in face:incident[v].append(i)
    lines=["# Cup workcell visual LOD, derived from FURRY's licensed G1 exterior.\n"]
    lines.extend("v %.9g %.9g %.9g\n"%tuple(v) for v in vertices)
    ids=[]
    for i,face in enumerate(faces):
        corner=[]
        for v in face:
            neighbors=normals[incident[v]]
            smooth=neighbors[neighbors@normals[i]>np.cos(np.deg2rad(40))].sum(axis=0)
            smooth/=max(np.linalg.norm(smooth),1e-16)
            lines.append("vn %.9g %.9g %.9g\n"%tuple(smooth));corner.append(len(ids)*3+len(corner)+1)
        ids.append(corner)
    for face,ns in zip(faces,ids):lines.append("f "+" ".join(f"{v+1}//{n}" for v,n in zip(face,ns))+"\n")
    path.write_text("".join(lines))


def main():
    DEST.mkdir(exist_ok=True);records={}
    for name,target in [("torso_link",5800),("head_link",1600)]:
        source=SOURCE/(name+".obj");v,f=load(source)
        reduced,faces=fast_simplification.simplify(v,f,target_count=target,agg=5)
        samples=np.r_[v,v[f[::7]].mean(axis=1)]
        reverse=np.r_[reduced,reduced[faces[::7]].mean(axis=1)]
        distances=np.r_[distance(samples,reduced[faces]),distance(reverse,v[f])]
        if distances.max()>.003:raise ValueError(f"LOD exceeds 3 mm sampled surface error: {name}: {distances.max()}")
        output=DEST/("cup-lab-"+name+".obj");write(output,reduced,faces)
        records[name]={"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                       "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
                       "source_triangles":len(f),"triangles":len(faces),
                       "sampled_bidirectional_surface_error_max_m":float(distances.max()),
                       "surface_samples":len(distances)}
    (DEST/"manifest.json").write_text(json.dumps({"method":"QEM; crease-aware corner normals; visual only", "meshes":records},indent=2)+"\n")


if __name__=="__main__":main()
