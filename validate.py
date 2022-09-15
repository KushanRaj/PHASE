import argparse

import torch
import numpy as np

import trimesh
from glob import glob
import os
from tqdm import tqdm

from Chamfer3D.dist_chamfer_3D import chamfer_3DDist


POINTS = {'dragon' : 100_000, 'armadillo' : 50_000, 'bunny' : 50_000}

def norm(points):

    points -= points.mean(0, keepdims = True)

    points /= points.pow(2).sum(-1).pow(0.5).max()

    return points

class ChamferLoss:
    def __init__(self):
        self.kernel = chamfer_3DDist()

    def __call__(self, pc1, pc2):
        dist1, dist2, _, _ = self.kernel(pc1, pc2)
        return torch.mean(dist1) + torch.mean(dist2)

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()))
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh

def sample_mesh(m, n):
    vpos, _ = trimesh.sample.sample_surface(m, n)
    return torch.tensor(vpos, dtype=torch.float32, device="cuda")

def keys(f):

    return int(f.split('/')[-1].split('_')[0])


if __name__ == "__main__":
    with torch.no_grad():
        chamfer_dist = ChamferLoss()

        for TYPE in ['dragon', 'armadillo', 'bunny']:

            print(f'{TYPE}')

            best_folder = ''
            best_loss = float('inf')
            best_file = ''

            store = []

            ref = as_mesh(trimesh.load(f'data/meshes/{TYPE}.obj'))

            vpos_ref = norm(sample_mesh(ref, POINTS[TYPE]))

            for folder in [i for i in os.listdir('output') if TYPE in i]:
                print(folder)
                
                for ind, file in enumerate(tqdm(sorted(list(glob(f'output/{folder}/*.ply')), key = keys))):

                    if ind > 20:
                        break
                
                    mesh = as_mesh(trimesh.load(file))
                    
                    vpos_mesh = norm(sample_mesh(mesh, POINTS[TYPE]))

                    loss = chamfer_dist(vpos_mesh[None, ...], vpos_ref[None, ...]).item()

                    store.append([folder, file, loss])
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_folder = folder
                        best_file = file

            print(best_folder, best_file, best_loss)
            
            
            data = np.array(store)
            np.save(f'saved_{TYPE}.npy', data[np.argsort(data[..., -1], )])
