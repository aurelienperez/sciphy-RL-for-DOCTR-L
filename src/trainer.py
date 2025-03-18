#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:21:31 2025

@author: aurelienperez
"""

import torch
from tqdm import trange

class Trainer():
    def __init__(self, model, dataset, loss_function, optimizer, num_epochs, batch_size, env,rng = None,force_training=False, scheduler=None,
                 patience=5, min_delta=1e-4, ):
        """
        Agent d'entrainement du modèle
        """
        if rng is None:
            rng = torch.generator(env.device)
        self.model = model
        self.dataloader = dataset.get_dataloader(batch_size)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.env = env
        self.policy = env.policy
        self.scheduler = scheduler
        self.history = {"epoch": [], "loss": [], "lr": []}
        self.device = env.device

        # Nombre d'epochs sans amélioration avant arrêt
        self.patience = patience  
        # Seuil minimal d'amélioration à considérer
        self.min_delta = min_delta  
        # Forcer à aller jusqu'au bout ?
        self.force_training = force_training  
        # Meilleure loss observée
        self.best_loss = float('inf')  
        # Compteur d'epochs sans amélioration
        self.epochs_no_improve = 0  

    def train(self):
        """"
        Méthode pour l'entrainement du modèle'
        """
        print(f"Starting training for {self.num_epochs} epochs")
        with trange(self.num_epochs) as pbar:
            for epoch in pbar:
                total_loss = 0.0
    
                for X, C, t in self.dataloader:
                    X, C, t = X.to(self.device), C.to(self.device), t.to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.loss_function(self.model, self.env, X, C, t)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
    
                avg_loss = total_loss / len(self.dataloader)
    
                # Mise à jour du scheduler
                if self.scheduler:
                    self.scheduler.step()
                    self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
    
                # Enregistrement de la loss
                self.history["epoch"].append(epoch)
                self.history["loss"].append(avg_loss)
    
                pbar.set_postfix(loss=avg_loss)
    
                # Vérification de l'amélioration de la loss
                if avg_loss < self.best_loss - self.min_delta:
                    self.best_loss = avg_loss
                    self.epochs_no_improve = 0  # Réinitialisation du compteur
                else:
                    self.epochs_no_improve += 1  # Incrémentation si pas d'amélioration
    
                # Vérification du critère d'arrêt (si pas forcé)
                if not self.force_training and self.epochs_no_improve >= self.patience:
                    print(f"Arrêt anticipé après {epoch + 1} epochs (aucune amélioration significative depuis {self.patience} epochs).")
                    break
