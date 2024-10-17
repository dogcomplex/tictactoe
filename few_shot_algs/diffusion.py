import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from few_shot_algs.few_shot_alg import Algorithm
import random

class DiffusionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, t):
        x = torch.cat([x, t.unsqueeze(1)], dim=1)
        return self.net(x)

class DiffusionAlgorithm(Algorithm):
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=5, num_timesteps=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps

        # Diffusion model
        self.model = DiffusionNet(input_dim + 1, hidden_dim, input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

        # Diffusion parameters
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Memory for few-shot learning
        self.memory = []

        # Add this line to specify the number of classes
        self.num_classes = 5

    def predict(self, observation: str) -> int:
        x = self.state_to_tensor(observation).to(self.device)
        x = self.denoise(x)
        # Map the denoised output to a valid label
        return self.map_to_label(x)

    def map_to_label(self, x):
        # Take only the first 5 elements (corresponding to the 5 classes)
        class_scores = x[0, :self.num_classes]
        # Return the index of the highest score
        return torch.argmax(class_scores).item()

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        self.memory.append((self.state_to_tensor(observation), correct_label))
        if len(self.memory) > 1000:  # Increased memory size
            self.memory.pop(0)
        self.train_step()

    def train_step(self):
        if len(self.memory) < 10:  # Wait until we have enough samples
            return

        for _ in range(5):  # Perform multiple training iterations
            batch = random.sample(self.memory, min(32, len(self.memory)))
            x, _ = zip(*batch)
            x = torch.cat(x).to(self.device)

            t = torch.randint(0, self.num_timesteps, (x.shape[0],)).to(self.device)
            noise = torch.randn_like(x)
            noisy_x = self.add_noise(x, t, noise)
            predicted_noise = self.model(noisy_x, t)

            loss = nn.MSELoss()(noise, predicted_noise)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

    def add_noise(self, x, t, noise):
        alphas_cumprod_t = self.alphas_cumprod[t].unsqueeze(-1)
        return torch.sqrt(alphas_cumprod_t) * x + torch.sqrt(1 - alphas_cumprod_t) * noise

    def denoise(self, x):
        with torch.no_grad():
            for t in reversed(range(self.num_timesteps)):
                t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
                predicted_noise = self.model(x, t_tensor)
                alpha_t = self.alphas[t]
                alpha_t_cumprod = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod)) * predicted_noise) + torch.sqrt(beta_t) * noise
        return x

    @staticmethod
    def state_to_tensor(state: str) -> torch.Tensor:
        # Modify this method to output a tensor of shape (1, 9)
        return torch.tensor([[int(char) for char in state]], dtype=torch.float32)
