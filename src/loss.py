#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:20:46 2025

@author: aurelienperez
"""

import torch 

def hamiltonians(J_x, J_C, env, x, t):
    """
    Calcule les Hamiltoniens associés à la politique.
    """
    
    uk = env.policy.means(t, x)  # (batch_size, dimA, K)

    # Calcul de u^T u pour chaque composante  
    uTu = torch.einsum("bka,bak->bk", uk.mT, uk)  # (batch_size, K)

    µ1 = env.µ1(x)  # (batch_size, dimX, dimA)
    f1 = env.f1(x)  # (batch_size,)

    # Produit uTu avec f1 * J_C  
    uTuf1J_C = torch.einsum("bk,b->bk", uTu, f1 * J_C)  # (batch_size, K)

    # Contraction de µ1 et J_x  
    temp = torch.einsum("nij,ni -> nj", µ1, J_x)  # (batch_size, dimA)

    # Contraction de uk avec temp  
    uTµ1J_x = torch.einsum("bka,ba->bk", uk.mT, temp)  # (batch_size, K)

    sigma2 = env.policy.variances(t, x)  # (batch_size, dimA, K)

    # Élément carré de temp  
    temp2 = temp ** 2  # (batch_size, dimA)

    # Contraction de sigma2 avec temp2  
    lambdasigma2µ1J_x2 = 1/env.lambd * torch.einsum("bka,ba->bk", sigma2.mT, temp2)  # (batch_size, K)

    # Calcul du numérateur  
    numerateur = 0.5 * uTuf1J_C + uTµ1J_x - 0.5 * lambdasigma2µ1J_x2  # (batch_size, K)

    # Calcul du dénominateur  
    term = torch.einsum("bka,b->bk", sigma2.mT, f1 * J_C)
    term = torch.clamp(term, min=-env.lambd + 1e-6)
    denom = 1 + 1/env.lambd * term # (batch_size, K)

    # Deuxième terme du Hamiltonien  
    deuxieme_terme = env.lambd / 2 * torch.log(denom)  # (batch_size, K)

    # Hamiltoniens  
    return numerateur / denom + deuxieme_terme  # (batch_size, K)

def uk_J(J_x, J_C, env, x, t):
    """
    Calcule les moyennes des composantes gaussiennes de la politique optimale.
    """

    # Variances des composantes gaussiennes de la politique comportementale
    sigma_k2 = env.policy.variances(t, x)  # (batch_size, dimA, K)

    # Moyennes des composantes gaussiennes de la politique comportementale
    uk_x_t = env.policy.means(t, x)  # (batch_size, dimA, K)

    # Matrice de contrôle dans la dynamique
    mu1_x_t = env.µ1(x)  # (batch_size, dimX, dimA)

    # Coût quadratique
    f1_x_t = env.f1(x)  # (batch_size,)

    # Calcul du terme βσ²_k μ_1(x_t) · ∂J/∂x_t
    lambd_sigma2_mu1J_x = 1/env.lambd * torch.einsum("bak,bxa,bx->bak", sigma_k2, mu1_x_t, J_x)  # (batch_size, dimA, K)

    # Calcul du dénominateur : 1 + βσ²_k c_1(x_t) ∂J/∂C_t
    denom = 1 + 1/env.lambd * torch.einsum("bak,b->bak", sigma_k2, f1_x_t * J_C)  # (batch_size, dimA, K)

    # Moyennes des composantes gaussiennes de la politique optimale
    uk_J_result = (uk_x_t - lambd_sigma2_mu1J_x) / denom  # (batch_size, dimA, K)

    return uk_J_result  # (batch_size, dimA, K)


def expected_action(J_x, J_C, env, x, t):
    """
    Calcule l'action espérée ⟨a_t⟩[J].
    """

    # Hamiltoniens pour chaque composante k  
    H_k = hamiltonians(J_x, J_C, env, x, t)  # (batch_size, K)

    # Poids normalisés des composantes du mélange  
    exp_term = torch.exp(torch.clamp(-1/env.lambd * H_k, min=-50, max=50))
    weights_k = exp_term * env.policy.weights  # (batch_size, K)
    w_k = weights_k / torch.sum(weights_k, dim=-1, keepdim=True)  # (batch_size, K)
    
    adjusted_u_k = uk_J(J_x, J_C, env, x, t)

    # Moyenne pondérée des actions  
    aJ = torch.einsum("bk,bak->ba", w_k, adjusted_u_k)  # (batch_size, dimA)

    return aJ  # (batch_size, dimA)



def hamiltonian(HkJ, atJ, env, J, J_x, x, x_next, dt):
    """
    Calcule l'Hamiltonien.
    """
    atJ = torch.zeros_like(atJ) ##### Erreur dans le papier
    lambd = env.lambd
    weights = env.policy.weights
    µ0 = env.µ0(x)  # (batch_size, dimX)
    µ1 = env.µ1(x)  # (batch_size, dimX, dimA)

    # Poids normalisés du mélange 
    exp_term = torch.exp(torch.clamp(-1/lambd * HkJ, min=-50, max=50))
    wexp = weights * exp_term  # (batch_size, K)

    # Terme logarithmique de normalisation  
    log_term = lambd * torch.log(torch.sum(wexp, dim=1)).unsqueeze(-1)  # (batch_size, 1)

    # Terme d'actualisation  
    rJ = env.r * J  # (batch_size, 1)

    # Différence entre état suivant et dynamique déterministe sous atJ  
    diff = (x_next - x)/ dt - µ0 - torch.einsum("bxa,ba->bx", µ1, atJ)  # (batch_size, dimX)

    # Produit scalaire avec J_x  
    diff_J_x = torch.einsum("bx,bx->b", diff, J_x).unsqueeze(-1)  # (batch_size, 1)

    # Retourne le Hamiltonien
    return log_term + diff_J_x + rJ  # (batch_size, 1)


def delta_S(J_x, J_C, aJ, env, x, x_next, t, dt):
    """
    Calcule ΔS le log-ratio de vraisemblance entre les transitions d'état sous les politiques optimale et comportementale.
    """

    # Actions sous la politique comportementale  
    a0 = torch.einsum("bak,k->ba", env.policy.means(t, x), env.policy.weights)  # (batch_size, dimA)

    # Récupération des paramètres  
    µ1 = env.µ1(x)  # (batch_size, dimX, dimA)
    sigma = torch.diagonal(env.vol(t, x), dim1=-2, dim2=-1)  # (batch_size, dimX)
    sigma2 = sigma ** 2  # (batch_size, dimX)

    # Numérateur : différence des actions propagée par μ1  
    numerator = torch.einsum("bxa,ba->bx", µ1, (aJ - a0))  # (batch_size, dimX)

    # Normalisation par la variance  
    fraction_term = numerator / sigma2  # (batch_size, dimX)
 
    second_term = (env.µ0(x) + 0.5 * torch.einsum("bxa,ba->bx", µ1, (aJ + a0))) * dt - x_next + x  # (batch_size, dimX)

    # Somme sur les dimensions pour obtenir un scalaire par batch  
    delta_S_result = torch.einsum("bx,bx->b", fraction_term, second_term)  # (batch_size,)

    return delta_S_result  # (batch_size,)


def loss_function(model, env, X, C, t):
    """
    Calcule la perte qui est proportionnelle à l'opposé de la log-vraisemblance d'un batch de trajectoires
    """
    batch_size, n_steps, dimX = X.shape
    dt = env.T / n_steps  # Pas de temps

    loss_eq = 0  # Initialisation de la loss

    for i in range(n_steps - 1):  # Boucle sur les instants t
        x_, C_, t_ = X[:, i, :], C[:, i], t[:, i]  # (batch_size, dimX), (batch_size,), (batch_size,)
        x_next, C_next, t_next = X[:, i+1, :], C[:, i+1], t[:, i+1]  # (batch_size, dimX), (batch_size,), (batch_size,)

        # Évaluation de J_θ aux temps t et t+1
        J_t = model(x_, C_, t_)  # (batch_size,)
        J_t_next = model(x_next, C_next, t_next)  # (batch_size,)

        # Gradients de J par rapport à x et C
        J_x, J_C = model.compute_gradients(x_, C_, t_)  # J_x (batch_size, dimX), J_C (batch_size,)
        # J_x, J_C = model.compute_gradients(x_next, C_next, t_next)
        # Condition terminale
        if i == n_steps - 2:
            J_t_next = env.U(C[:, -1])  # (batch_size,)

        # Calcul des Hamiltoniens H_k[J]
        HkJ = hamiltonians(J_x, J_C, env, x_, t_)  # (batch_size, K)

        # Calcul de l'action attendue sous J
        atJ = expected_action(J_x, J_C, env, x_, t_)  # (batch_size, dimA)

        # Calcul du Hamiltonien H_HJ
        H_HJ = hamiltonian(HkJ, atJ, env, J_t, J_x, x_, x_next, dt)  # (batch_size, 1)
        H_HJ = H_HJ.squeeze(-1)  # (batch_size,)

        # Calcul de ΔS
        deltaS = delta_S(J_x, J_C, atJ, env, x_, x_next, t_, dt)  # (batch_size,)

        # Ajout de la perte backward
        loss_eq += torch.mean(0.5 * ((J_t_next - J_t) - H_HJ * dt) ** 2 + env.nu_2 * deltaS)

    return loss_eq  # (scalar)



