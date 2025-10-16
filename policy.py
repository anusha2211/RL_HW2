import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_POSITION = -1.2
MAX_POSITION = 0.5
MIN_VELOCITY = -0.07
MAX_VELOCITY = 0.07
TIMEOUT      = 1000


# Policy Network (PyTorch)
class PolicyNet(nn.Module):
    """
    Policy neural network:
      - Inputs:   2 numbers corresponding to the current state [x, v]
      - Outputs:  3 score values ("preferences"), one per action: Pref^{Reverse}, Pref^{Neutral}, Pref^{Forward}
      - The 'act' method takes as input a state, computes the 3 scores values ("preferences") mentioned above, and
        transforms them into probabilities via a Softmax operation. It then returns a_t = -1.0 if the sampled action is Reverse,
        a_t = 0.0 is the sampled action is Neutral, and a_t = +1 if the sampled action is Forward.
    """

    def __init__(self, neurons_per_layer):
        super().__init__()
        layers = []
        n_inputs  = 2 # Number of inputs  is 2 because the state is s=[x,v])
        n_outputs = 3 # Number of outputs is 3 because the network outputs one score value ("preference") for each action
        self.temperature = 0.1

        last = n_inputs 
        for h in neurons_per_layer:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, n_outputs))
        self.net = nn.Sequential(*layers)
        
        # Precompute constants needed to later perform [-1, 1] normalization when given an unnormalized state s=[x,v] as input
        self.pos_mid  = 0.5 * (MIN_POSITION + MAX_POSITION)
        self.pos_half = 0.5 * (MAX_POSITION - MIN_POSITION)
        self.vel_mid  = 0.5 * (MIN_VELOCITY + MAX_VELOCITY)
        self.vel_half = 0.5 * (MAX_VELOCITY - MIN_VELOCITY)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[..., 0]
        vel = x[..., 1]
        pos_n = (pos - self.pos_mid) / self.pos_half
        vel_n = (vel - self.vel_mid) / self.vel_half
        return torch.stack([pos_n, vel_n], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self._normalize(x)
        return self.net(x_n)

    @torch.no_grad()
    def act(self, state_np: np.ndarray) -> float:
        x = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0)  # (1,2)
        logits = self.forward(x).squeeze(0)  # (3,)
        probs = F.softmax(logits / self.temperature, dim=-1).cpu().numpy()
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        action_idx = int(np.random.choice(3, p=probs))
        a_map = (-1.0, 0.0, 1.0)
        return a_map[action_idx]
    
    def get_policy_parameters(self) -> np.ndarray:
        with torch.no_grad():
            return torch.cat([p.view(-1) for p in self.parameters()]).cpu().numpy().copy()

    def set_policy_parameters(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=np.float32)
        idx = 0
        with torch.no_grad():
            for p in self.parameters():
                numel = p.numel()
                block = torch.from_numpy(theta[idx:idx+numel]).view_as(p)
                p.copy_(block)
                idx += numel
        assert idx == theta.size, "Length of vector 'theta' does not match number of policy parameters"

