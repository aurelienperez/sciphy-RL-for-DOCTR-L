#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:19:03 2025

@author: aurelienperez
"""

import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    """ Approximation de la fonction valeur J(x, C, t). """

    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        """ Initialise le réseau. """
        super(ValueNetwork, self).__init__()
        layers = [nn.Linear(dim_in, dim_hidden), nn.Softplus()]  # (dim_in -> dim_hidden)

        # Couches cachées  
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim_hidden, dim_hidden))  # (dim_hidden -> dim_hidden)
            layers.append(nn.Softplus())

        # Dernière couche linéaire  
        layers.append(nn.Linear(dim_hidden, dim_out))  # (dim_hidden -> dim_out)
        # layers.append(nn.Softplus())  # Sortie positive
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x, C, t):
        """ Évalue J(x, C, t). """
        # Concatène les entrées  
        res = torch.cat([x, C.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)  # (batch_size, dimX + 2)
        # Passage dans les couches  
        for layer in self.layers:
            res = layer(res)

        return res  # (batch_size, dim_out)

    def compute_gradients(self, x, C, t):
        """ Calcule dJ/dx et dJ/dC. """
        # Active la différentiabilité sur x et C  
        x.requires_grad_(True)  # (batch_size, dimX)
        C.requires_grad_(True)  # (batch_size,)

        # Évalue J(x, C, t)  
        J = self.forward(x, C, t)  # (batch_size, 1)

        # Gradient spatial  
        J_x = torch.autograd.grad(J, x, grad_outputs=torch.ones_like(J), retain_graph=True, create_graph=True)[0]  # (batch_size, dimX)

        # Gradient par rapport à C  
        J_C = torch.autograd.grad(J, C, grad_outputs=torch.ones_like(J), retain_graph=True, create_graph=True)[0]  # (batch_size,)

        return J_x, J_C  # (batch_size, dimX), (batch_size,)
