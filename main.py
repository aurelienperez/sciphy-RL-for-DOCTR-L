import torch

from src.utils import set_device, plot_some_trajectories, plot_loss_lr, init_weights, plot_uk_vs_xi
from src.policy import Policy
from src.environment import Environment
from src.simulation import simulate_data
from src.dataset import TrajectoryDataset
from src.value_network import ValueNetwork
from src.loss import loss_function
from src.trainer import Trainer

device = set_device()
print(f"Using device: {device}")

rng = torch.Generator(device=device)

K = 2  # Nombre de gaussiennes dans le mélange
dimX = 100
dimA = 5
lambd = 0.2
r = 0.03
nu_2 = 10
T = 1

U = lambda x: 1/2 * x**2


######### Policy ###############
# Initialiser des valeurs fixes pour la politique comportementale
fixed_means = (torch.rand((dimA, K), generator=rng, dtype=torch.float32, device=device) - 0.5)  # (dimA, K)
fixed_variances = (torch.rand((dimA, K), generator=rng, dtype=torch.float32, device=device) * 0.2 + 0.2)  # (dimA, K)

# Définition de la politique comportementale
means = lambda t, x: fixed_means.expand(x.shape[0], -1, -1)  # (batch_size, dimA, K)
variances = lambda t, x: fixed_variances.expand(x.shape[0], -1, -1)  # (batch_size, dimA, K)


policy = Policy(means_func = means, variances_func = variances, dimA = dimA, K = K, device = device)


######### Environment ##########

#### Drift ####
# Terme constant du drift  
µ00 = 0.01 * torch.ones((dimX,), device=device)  # (dimX,)

# Matrice de couplage état-état dans le drift  
µ01 = 0.02 * torch.eye(dimX, device=device)  # (dimX, dimX)

# Matrice de couplage état-action dans le drift (terme constant)  
µ10 = 0.01 * torch.eye(dimX, dimA, device=device)  # (dimX, dimA)

# Matrice de couplage état-action dans le drift (terme linéaire)  
µ11 = 0.02 * torch.eye(dimX, dimA, device=device)  # (dimX, dimA)

### Volatilité ###
vol = lambda t, x: torch.diag_embed(0.4 * x)  # (batch_size, dimX, dimX)

### Coût ###
# Fonction coût quadratique de l'état  
f0 = lambda x: torch.norm(x, dim=1) ** 2  # (batch_size,)

# Poids du terme quadratique dans le coût  
f1 = lambda x: 5.0 * torch.norm(x, dim=1) ** 2  # (batch_size,)

env = Environment(policy = policy, lambd = lambd, r = r, nu_2 = nu_2, dimX = dimX, T = T, U = U, µ00 = µ00, µ01 = µ01, µ10 = µ10, µ11 = µ11, vol = vol, f0 =f0, f1 = f1)



######### Simulation ###########
n_steps, n_traj = 40, 1_000


# Génération des trajectoires
X0 = (torch.rand((n_traj, env.dimX), device=env.device) + 1) / 10  # (n_traj, dimX)
Y = simulate_data(n_traj = n_traj, n_steps = n_steps, X0 = X0, env = env, rng = rng) # (n_traj, n_steps, dimX + 1)

X, C = Y[:,:,:-1],  Y[:,:,-1]

rc = plot_some_trajectories(X, env)

######### Entrainement ###########

# Création du DataLoader avec des batchs de trajectoires complètes
dataset = TrajectoryDataset(Y, env.T)
batch_size = 256

# Définition du modèle et de l'optimiseur
n_epochs = 30
dim_in = env.dimX + 2  # Dimension d'entrée du réseau : dimX + coût accumulé + temps

model = ValueNetwork(dim_in=dim_in, dim_out=1, dim_hidden=100, num_layers=3).to(device)
model.apply(lambda m:init_weights(m, rng))
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, total_steps=n_epochs, pct_start=0.3)
# scheduler = None

# Instanciation et entraînement
trainer = Trainer(model, dataset, loss_function, 
                  optimizer, num_epochs=n_epochs, batch_size = batch_size, env=env, 
                  scheduler=scheduler, rng = rng,force_training=True)


trainer.train() 

plot_loss_lr(trainer)

rang = 10
plot_uk_vs_xi(model, env, Y[:,:,:-1], Y[:,:,-1], dim_x=rc, num_points=100,rang = rang, policy_type = "behavioural")
plot_uk_vs_xi(model, env, Y[:,:,:-1], Y[:,:,-1], dim_x=rc, num_points=100,rang = rang, policy_type = "optimal")
