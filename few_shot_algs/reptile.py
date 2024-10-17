import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from few_shot_algs.few_shot_alg import Algorithm

class SimpleNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MAMLReptileAlgorithm(Algorithm):
    def __init__(self, input_size: int = 9, hidden_size: int = 27, output_size: int = 5, 
                 lr: float = 0.005, inner_lr: float = 0.1, inner_steps: int = 3):
        super().__init__()
        self.model = SimpleNet(input_size, hidden_size, output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.loss_fn = nn.CrossEntropyLoss()

    def predict(self, observation: str) -> int:
        x = torch.tensor([int(char) for char in observation], dtype=torch.float32)
        with torch.no_grad():
            output = self.model(x)
        return output.argmax().item()

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        self._update()

    def _update(self):
        if len(self.history) < 2:  # Need at least 2 samples for train/test split
            return

        # Prepare data
        train_data = self.history[:-1]
        test_data = [self.history[-1]]

        # Inner loop (task-specific adaptation)
        fast_weights = [p.clone().detach() for p in self.model.parameters()]
        for _ in range(self.inner_steps):
            for obs, _, label in train_data:
                x = torch.tensor([int(char) for char in obs], dtype=torch.float32)
                y = torch.tensor(label, dtype=torch.long)
                logits = self.model(x)
                loss = self.loss_fn(logits.unsqueeze(0), y.unsqueeze(0))
                grads = torch.autograd.grad(loss, self.model.parameters())
                fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]

        # Outer loop (meta-update)
        self.optimizer.zero_grad()
        for p, w in zip(self.model.parameters(), fast_weights):
            p.grad = p.data - w.data
        self.optimizer.step()
