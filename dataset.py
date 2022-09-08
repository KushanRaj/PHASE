import torch
from torch.utils.data import Dataset
import numpy as np
import os


class PCLoader(Dataset):

    def __init__(self, root, name : str, points_batch=16384, with_normals=False):

        self.points_batch = points_batch
        self.with_normals = with_normals

        self.data = torch.from_numpy(np.loadtxt(f"{root}/{name}", delimiter=' ').astype(np.float32)) 

        if self.with_normals:
            self.normals = self.data[:, 3:]
            self.data = self.data[:, :3]   


    def __getitem__(self, index):


        idx = torch.randperm(self.data.shape[0])[:self.points_batch]
        points = self.data[idx]

        points -= points.mean(0, keepdims = True)

        points /= points.pow(2).sum(-1).pow(0.5).max()
    
        if self.with_normals:
            normals = self.normals[idx]

        else:
            normals = torch.empty(0)

        phi = (((torch.rand(points.shape) * (points.max(0)[0] - points.min(0)[0])) + points.min(0)[0]).numpy()) * 1.5

        return points, normals, phi

    def __len__(self):
        return len(self.data)

