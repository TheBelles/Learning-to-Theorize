import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    """Applies Feature-wise Linear Modulation (FiLM)."""
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        # Outputs both gamma (scale) and beta (shift)
        self.fc = nn.Linear(cond_dim, feature_dim * 2)
        
    def forward(self, x, condition):
        gamma_beta = self.fc(condition)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return (1 + gamma) * x + beta

class CNNEncoder(nn.Module):
    """Maps 3x32x32 image observations into a latent state."""
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class CNNDecoder(nn.Module):
    """Maps latent states back to 3x32x32 image observations."""
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # -> 32x32
            nn.Sigmoid() # Constrain outputs to [0, 1] for valid image pixels
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        return self.net(x)

class Programmer(nn.Module):
    """Generates the next primitive operation using FiLM conditioning."""
    def __init__(self, latent_dim, num_primitives):
        super().__init__()
        self.film = FiLMLayer(cond_dim=latent_dim, feature_dim=latent_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_primitives)
        )

    def forward(self, current_state, target_state):
        # The target state modulates the current state representation
        modulated_state = self.film(current_state, target_state)
        logits = self.fc(modulated_state)
        return logits

class Executor(nn.Module):
    """Applies the chosen primitive to the current state using FiLM."""
    def __init__(self, latent_dim, num_primitives):
        super().__init__()
        self.primitive_embeddings = nn.Linear(num_primitives, latent_dim, bias=False)
        self.film = FiLMLayer(cond_dim=latent_dim, feature_dim=latent_dim)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, state, primitive_onehot):
        # Embed the discrete primitive choice
        z_emb = self.primitive_embeddings(primitive_onehot)
        # The chosen action modulates the current state
        modulated_state = self.film(state, z_emb)
        next_state = self.transition(modulated_state)
        return next_state

class NeuralTheorizer(nn.Module):
    def __init__(self, latent_dim=256, num_primitives=8, max_steps=3):
        super().__init__()
        self.max_steps = max_steps
        
        self.encoder = CNNEncoder(latent_dim)
        self.decoder = CNNDecoder(latent_dim)
        self.programmer = Programmer(latent_dim, num_primitives)
        self.executor = Executor(latent_dim, num_primitives)

    def forward(self, x, y, tau=1.0, hard=True):
        s_t = self.encoder(x)
        s_target = self.encoder(y)
        
        intermediate_states = [s_t]
        intermediate_reconstructions = []
        primitive_choices = []
        
        for step in range(self.max_steps):
            logits = self.programmer(s_t, s_target)
            z_t = F.gumbel_softmax(logits, tau=tau, hard=hard)
            primitive_choices.append(z_t)
            
            s_t = self.executor(s_t, z_t)
            intermediate_states.append(s_t)
            
            y_hat = self.decoder(s_t)
            intermediate_reconstructions.append(y_hat)
            
        return {
            "reconstructions": intermediate_reconstructions, 
            "primitives": primitive_choices,
            "final_state": s_t,
            "target_state": s_target
        }

    def compute_loss(self, x, y):
        outputs = self(x, y)
        reconstructions = outputs["reconstructions"]
        
        losses = []
        state_grounding_losses = []
        
        for y_hat in reconstructions:
            # 1. Target Reconstruction Loss
            # Using MSE over the image spatial dimensions (C, H, W)
            step_loss = F.mse_loss(y_hat, y, reduction='none').view(y.size(0), -1).mean(dim=-1)
            losses.append(step_loss)
            
            # 2. State Grounding Loss (Total Variation)
            # Ensures intermediate states decode into smooth, continuous images, 
            # anchoring the latent space representations to the image manifold.
            tv_loss = torch.mean(torch.abs(y_hat[:, :, :, :-1] - y_hat[:, :, :, 1:])) + \
                      torch.mean(torch.abs(y_hat[:, :, :-1, :] - y_hat[:, :, 1:, :]))
            state_grounding_losses.append(tv_loss)
            
        losses = torch.stack(losses, dim=1) 
        min_loss, optimal_steps = torch.min(losses, dim=1)
        
        length_penalty = (optimal_steps.float() * 0.01)
        avg_grounding_loss = torch.stack(state_grounding_losses).mean() * 0.1
        
        total_loss = (min_loss + length_penalty).mean() + avg_grounding_loss
        return total_loss