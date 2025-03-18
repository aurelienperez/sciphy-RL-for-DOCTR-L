#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:23:48 2025

@author: aurelienperez
"""

import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    """
    Dataset de trajectoires d'états et de coûts.
    """

    def __init__(self, Y, T):
        """Initialise les données."""
        self.X = Y[:, :, :-1]  # (n_traj, n_steps, dimX)
        self.C = Y[:, :, -1]  # (n_traj, n_steps)
        self.T = T  
        self.n_traj, self.n_steps, self.dimX = self.X.shape  # Dimensions du dataset

    def __len__(self):
        """Retourne le nombre de trajectoires."""
        return self.n_traj  

    def __getitem__(self, idx):
        """Retourne une trajectoire donnée."""
        x_traj = self.X[idx]  # (n_steps, dimX)
        C_traj = self.C[idx]  # (n_steps,)
        t_traj = torch.linspace(0, self.T, steps=self.n_steps, device=self.X.device)  # (n_steps,)

        return x_traj, C_traj, t_traj
    
    def get_dataloader(self,batch_size):
        """Retourne un distributeur de batch."""
        return DataLoader(self, batch_size=batch_size, shuffle=True)