"""Export authored visuals without decimating again or discarding corner normals."""
import struct
import numpy as np


def mesh_stream(model, mesh_id):
    va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    na = model.mesh_normaladr[mesh_id]
    points = model.mesh_vert[va:va+vn][model.mesh_face[fa:fa+fn]]
    normals = model.mesh_normal[na + model.mesh_facenormal[fa:fa+fn]]
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError('Non-finite visual mesh data')
    return np.concatenate([points, normals], axis=2).astype('<f4')


def export_visual_meshes(model, path, visible_mesh_ids):
    counts = [0] * model.nmesh
    with path.open('wb') as stream:
        stream.write(struct.pack('<I', model.nmesh))
        for i in range(model.nmesh):
            packed = mesh_stream(model, i) if i in visible_mesh_ids else np.empty((0, 3, 6), dtype='<f4')
            counts[i] = len(packed)
            stream.write(struct.pack('<I', len(packed)*3))
            stream.write(packed.tobytes())
    return counts
