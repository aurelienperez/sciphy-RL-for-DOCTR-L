#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:18:07 2025

@author: aurelienperez
"""

import torch


class Environment:
    """
    Classe définissant les paramètres et l'état du modèle.
    """

    def __init__(self, policy, lambd, r, nu_2, dimX, T, U, µ00, µ01, µ10, µ11, vol, f0, f1):
        # Politique comportementale  
        self.policy = policy  
        self.device = policy.device  

        # Température (régularisation KL) 
        # Plus la température est élevée, plus on est proche de la politique comportementale
        self.lambd = lambd # Scalaire  

        # Taux d'actualisation  
        self.r = r  # Scalaire  

        # Variance de l'erreur de regression
        self.nu_2 = nu_2  # Scalaire  

        # Dimension de l'espace d'état  
        self.dimX = dimX  # Scalaire  

        # Dimension de l'espace d'action  
        self.dimA = policy.dimA  # Scalaire  

        # Horizon temporel  
        self.T = T  # Scalaire  

        # Nombre de composantes gaussiennes dans la politique  
        self.K = policy.K  # Scalaire  

        # Fonction de coût terminal (utilité convexe qui vaut 0 en 0)  
        self.U = U  

        ####### Coefficients du drift et des coûts (linéaire-quadratique) #######
        
        # Terme constant du drift  
        self.µ00 = µ00  # (dimX,)

        # Matrice de couplage état-état dans le drift  
        self.µ01 = µ01  # (dimX, dimX)

        # Matrice de couplage état-action dans le drift (terme constant)  
        self.µ10 = µ10  # (dimX, dimA)

        # Matrice de couplage état-action dans le drift (terme linéaire)  
        self.µ11 = µ11  # (dimX, dimA)

        # Fonction µ0: drift sans action  
        self.µ0 = lambda x: self.µ00 + torch.einsum("xy,by->by", self.µ01, x)  # (batch_size, dimX)

        # Fonction µ1: influence de l'action sur le drift  
        self.µ1 = lambda x: self.µ10 + torch.einsum("xa,bx->bxa", self.µ11, x)  # (batch_size, dimX, dimA)

        # Fonction coût quadratique de l'état  
        self.f0 = f0  # (batch_size,)

        # Poids du terme quadratique dans le coût  
        self.f1 = f1 # (batch_size,)

        # Drift du processus d'état (dynamique moyenne)  
        self.drift = lambda t, x: self.µ0(x) + torch.einsum("bxa,ba->bx", self.µ1(x), torch.mean(policy.means(t, x), axis=2))  # (batch_size, dimX)

        # Matrice de volatilité (ne dépend pas de l'action)  
        self.vol = vol  # (batch_size, dimX, dimX)
