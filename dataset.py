import torch
from torch.utils.data import Dataset
import numpy as np
import os


class PCLoader(Dataset):

    def __init__(self, root, name : str, points_batch=16384, with_normals=False, n_epochs = 6, batch_size = 6):

        self.points_batch = points_batch
        self.with_normals = with_normals
        self.batch_size = batch_size

        self.data = torch.from_numpy(np.loadtxt(f"{root}/{name}", delimiter=' ').astype(np.float32)) 

        self.n_epochs = n_epochs

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

        phi = (((torch.rand(points.shape) * (points.max(0)[0] - points.min(0)[0])) + points.min(0)[0])) * 1.5

        return points, normals, phi

    def __len__(self):
        return (1_000 * self.batch_size)

