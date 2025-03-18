#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:19:52 2025

@author: aurelienperez
"""

import torch
import numpy as np

def cost_function(x, a, env):
    """
    Calcule le coût instantané.
    """
    return env.f0(x) + 0.5 * env.f1(x) * torch.norm(a, dim=1) ** 2  # (batch_size,)

def simulate_data(n_traj, n_steps, X0, env, rng=None):
    """
    Simule des trajectoires pour l'environnement.
    """
    if rng is None:
        rng = torch.Generator(device=env.device)

    dt = env.T / n_steps  # Pas de temps


    dW = torch.randn(size=(n_traj, n_steps - 1, env.dimX), generator=rng, device=env.device) * np.sqrt(dt)  # (n_traj, n_steps-1, dimX)

    # États et coûts accumulés  
    X = torch.zeros((n_traj, n_steps, env.dimX), device=env.device)  # (n_traj, n_steps, dimX)
    C = torch.zeros((n_traj, n_steps), device=env.device)  # (n_traj, n_steps)

    # Initialisation des états  
    X[:, 0, :] = X0  # (n_traj, dimX)

    for t in range(n_steps - 1):
        # Action moyenne selon la politique  
        a = torch.einsum("bak,k->ba", env.policy.means(t * dt, X[:, t, :]), env.policy.weights)  # (n_traj, dimA)

        # Mise à jour de l'état  
        X[:, t + 1, :] = X[:, t, :] + env.drift(t * dt, X[:, t, :]) * dt + torch.einsum("bxy,by -> bx", env.vol(t * dt, X[:, t, :]), dW[:, t, :])  # (n_traj, dimX)

        # Mise à jour du coût accumulé  
        C[:, t + 1] = C[:, t] + (cost_function(X[:, t, :], a, env) + env.r * C[:, t]) * dt  # (n_traj,)

    return torch.cat([X, C.unsqueeze(-1)], dim=-1)  # (n_traj, n_steps, dimX + 1)
