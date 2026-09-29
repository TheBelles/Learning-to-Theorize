import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """Maps raw observations into a latent state."""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    """Maps latent states back to the observation space."""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class Programmer(nn.Module):
    """
    Acts as the 'Language of Thought' generator.
    Given the current state and target state, it predicts the next primitive operation.
    """
    def __init__(self, latent_dim, num_primitives):
        super().__init__()
        # Takes current latent state and target latent state
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_primitives)
        )

    def forward(self, current_state, target_state):
        x = torch.cat([current_state, target_state], dim=-1)
        logits = self.net(x)
        return logits

class Executor(nn.Module):
    """
    The shared transition model.
    Applies the chosen primitive to the current state to produce the next state.
    """
    def __init__(self, latent_dim, num_primitives):
        super().__init__()
        self.primitive_embeddings = nn.Linear(num_primitives, latent_dim, bias=False)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, state, primitive_onehot):
        # Embed the discrete primitive choice
        z_emb = self.primitive_embeddings(primitive_onehot)
        # Apply transition
        x = torch.cat([state, z_emb], dim=-1)
        next_state = self.transition(x)
        return next_state

class NeuralTheorizer(nn.Module):
    """
    The full NEO model implementing Learning-to-Theorize (L2T).
    """
    def __init__(self, input_dim, latent_dim=64, num_primitives=10, max_steps=5):
        super().__init__()
        self.max_steps = max_steps
        
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.programmer = Programmer(latent_dim, num_primitives)
        self.executor = Executor(latent_dim, num_primitives)

    def forward(self, x, y, tau=1.0, hard=True):
        """
        x: Source observation (before)
        y: Target observation (after)
        tau: Temperature for Gumbel-Softmax
        hard: Whether to use straight-through gradient estimation
        """
        batch_size = x.size(0)
        
        # 1. Encode source and target
        s_t = self.encoder(x)
        s_target = self.encoder(y)
        
        intermediate_states = [s_t]
        intermediate_reconstructions = []
        primitive_choices = []
        
        # 2. Induce latent program step-by-step
        for step in range(self.max_steps):
            # Programmer selects the next primitive
            logits = self.programmer(s_t, s_target)
            
            # Gumbel-Softmax allows differentiable sampling of discrete primitives
            z_t = F.gumbel_softmax(logits, tau=tau, hard=hard)
            primitive_choices.append(z_t)
            
            # Executor applies the primitive
            s_t = self.executor(s_t, z_t)
            intermediate_states.append(s_t)
            
            # Decode to check if we've reached the target
            y_hat = self.decoder(s_t)
            intermediate_reconstructions.append(y_hat)
            
        return {
            "reconstructions": intermediate_reconstructions, 
            "primitives": primitive_choices,
            "final_state": s_t,
            "target_state": s_target
        }

    def compute_loss(self, x, y):
        """
        Computes the L2T objective: 
        Finds the shortest latent program that successfully reconstructs the target.
        """
        outputs = self(x, y)
        reconstructions = outputs["reconstructions"]
        
        losses = []
        # Calculate reconstruction loss at each step
        for y_hat in reconstructions:
            step_loss = F.mse_loss(y_hat, y, reduction='none').mean(dim=-1)
            losses.append(step_loss)
            
        losses = torch.stack(losses, dim=1) # Shape: (batch_size, max_steps)
        
        # Adaptive Explanation Length: 
        # NEO selects the shortest accurate explanation. We can approximate this by 
        # taking the minimum loss across the trajectory (or stopping early when loss < threshold).
        min_loss, optimal_steps = torch.min(losses, dim=1)
        
        # We can also add a penalty for program length to encourage parsimony
        length_penalty = (optimal_steps.float() * 0.01)
        
        total_loss = (min_loss + length_penalty).mean()
        return total_loss