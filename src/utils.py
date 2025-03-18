#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:22:21 2025

@author: aurelienperez
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.loss import uk_J


def set_device():
    """
    Détermine l'appareil disponible pour les calculs (CPU, MPS, GPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def plot_some_trajectories(X, env):
    """
    Affiche quelques trajectoires d'une dimension aléatoire de l'état et retourne l'indice de la composante.
    """
    n_traj, n_steps, dimX = X.shape  
    
    # Sélection d'une seule dimension aléatoire de l'état
    random_component = np.random.choice(dimX, size=1).item()  
    
    # Affichage de 100 trajectoires pour cette composante
    for i in np.random.choice(n_traj, size=min(100, n_traj), replace=False):  # Sélection de 100 trajectoires (ou moins)
        plt.plot(np.arange(0, env.T, env.T / n_steps), X[i, :, random_component].cpu().numpy())  
    
    plt.title('Sample Trajectories for Random Component')  
    plt.xlabel('Time Steps')  
    plt.ylabel(f'Component {random_component}')  
    
    plt.show()  
    
    return random_component 
 


def plot_loss_lr(trainer):
    """
    Affiche l'évolution de la perte et du taux d'apprentissage au cours de l'entraînement.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 

    # Affichage de la perte au cours des epochs
    axes[0].plot(trainer.history["epoch"], trainer.history["loss"])  # (num_epochs,)
    axes[0].set_title('Loss Over Epochs')  
    axes[0].set_xlabel('Epochs')  
    axes[0].set_ylabel('Loss')  
    axes[0].legend(['Loss'])  

    # Affichage du taux d'apprentissage au cours des epochs
    axes[1].plot(trainer.history["epoch"], trainer.history["lr"])  # (num_epochs,)
    axes[1].set_title('Learning Rate Over Epochs')  
    axes[1].set_xlabel('Epochs') 
    axes[1].set_ylabel('Learning Rate')  
    axes[1].legend(['Learning Rate'])  

    plt.show()  


def get_typical_trajectory(X):
    """
    Sélectionne la trajectoire la plus proche de la médiane des normes L2 des trajectoires.
    """

    # Calcul de la norme L2 moyenne de chaque trajectoire sur le temps
    norms = torch.norm(X, dim=(1, 2))  # (n_traj,) - Norme L2 de chaque trajectoire

    # Trouver la médiane des normes
    median_norm = torch.median(norms).item()  # scalaire - Médiane des normes

    # Sélectionner la trajectoire la plus proche de la médiane
    idx_typical = torch.argmin(torch.abs(norms - median_norm)).item()  # scalaire - Indice de la trajectoire la plus proche

    # Retourner l’état x correspondant à cette trajectoire au dernier temps observé
    return X[idx_typical, 0, :]  # (dimX,) - L'état x au premier temps de la trajectoire sélectionnée

def init_weights(m, rng):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu', generator = rng)  # He init
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def plot_uk_vs_xi(model, env, X, C, dim_x, num_points, rang, policy_type='optimal'):
    """
    Affiche les moyennes $u_k[J]$ en fonction d'une composante de l'état, pour une politique optimale ou comportementale.
    """
    device = next(model.parameters()).device  

    # Génération des valeurs de x_{t,dim_x} variant de -rang à +rang
    x_range = torch.linspace(-rang, rang, num_points, device=device)  # (num_points,)

    # Sélection d'une trajectoire typique
    x_typical = get_typical_trajectory(X).to(device)  # (dimX,)
    x_fixed = x_typical.repeat(num_points, 1)  # (num_points, dimX)
    x_fixed[:, dim_x] = x_range  # (num_points, dimX) - Composante dim_x varie

    # Sélection d'une valeur typique de C
    C_typical = C[0, 0]  # Valeur fixe pour C
    C_fixed = torch.full((num_points,), C_typical, device=device)  # (num_points,)

    # Fixer la valeur de t
    t_fixed = torch.full((num_points,), 0, device=device)  # (num_points,)

    # Calcul des gradients de J_theta
    J_x, J_C = model.compute_gradients(x_fixed, C_fixed, t_fixed)  # (num_points, dimX), (num_points,)

    # Sélection de la politique : comportementale ou optimale
    if policy_type == 'behavioural':
        uk_values = env.policy.means(t_fixed, x_fixed)  # Politique comportementale, (num_points, dimA, K)
    else:  # Politique optimale
        uk_values = uk_J(J_x, J_C, env, x_fixed, t_fixed)  # (num_points, dimA, K)

    # Création dynamique des sous-plots selon le nombre de composantes du mélange
    fig, axes = plt.subplots(1, env.K, figsize=(6 * env.K, 5))

    # Si K == 1, plt.subplots renvoie un seul axe, on le met dans une liste
    if env.K == 1:
        axes = [axes]

    # Parcours des K composants du mélange gaussien
    for k in range(env.K):  # (env.K,)
        ax = axes[k]
        for d in range(env.dimA):  # (env.dimA,)
            ax.plot(x_range.cpu().numpy(), uk_values[:, d, k].detach().cpu().numpy(), label=f"dim {d+1}")

        ax.set_title(f"Means of $u_k[J]$ for GM component {k+1}")
        ax.set_xlabel("$x_{t,dim_x}$")
        ax.set_ylabel("$u_k[J]$")
        ax.legend()

    plt.tight_layout()  
    plt.show()  




