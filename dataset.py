import torch
from torch.utils.data import Dataset
import numpy as np
import os


class PCLoader(Dataset):

    def __init__(self, root, name : str, points_batch=16384, with_normals=False):

        self.points_batch = points_batch
        self.with_normals = with_normals

        if self.with_normals:
            data = torch.from_numpy(np.loadtxt(f"{root}/{name.replace('_', '_normals_')}", delimiter=' ').astype(np.float32))
            self.normals = data[:, 3:]
            self.data = data[:, :3]   

        else:
            self.data = torch.from_numpy(np.loadtxt(f"{root}/{name}", delimiter=' ').astype(np.float32)) 

        self.data /= self.data.pow(2).sum(-1).pow(0.5).max()


    def __getitem__(self, index):


        idx = torch.randperm(self.data.shape[0])[:self.points_batch]
        points = self.data[idx]
    
        if self.with_normals:
            normals = self.normals[idx]

        else:
            normals = torch.empty(0)

        return points, normals

    def __len__(self):
        return len(self.data)

