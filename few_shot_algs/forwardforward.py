import torch
import torch.nn as nn
import torch.optim as optim
from few_shot_algs.few_shot_alg import Algorithm
import random

class ForwardForwardAlgorithm(Algorithm, nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=5, num_layers=4, learning_rate=0.001, threshold=1.0, epochs=50, batch_size=16):
        Algorithm.__init__(self)
        nn.Module.__init__(self)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.threshold = threshold
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)
    
    def goodness(self, x):
        total_goodness = 0
        for layer in self.layers:
            x = layer(x)
            total_goodness += torch.sum(x ** 2, dim=1)
            x = torch.relu(x)
        return total_goodness
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self(x)
        loss = self.criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, observation: str) -> int:
        x = torch.tensor([int(digit) for digit in observation], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = self(x)
            prediction = torch.argmax(output, dim=1).item()
        
        return prediction
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        
        # Train on the entire history for multiple epochs
        if len(self.history) >= self.batch_size:
            for _ in range(self.epochs):
                random.shuffle(self.history)
                for i in range(0, len(self.history), self.batch_size):
                    batch = self.history[i:i+self.batch_size]
                    x = torch.tensor([[int(digit) for digit in obs] for obs, _, _ in batch], dtype=torch.float32)
                    y = torch.tensor([label for _, _, label in batch], dtype=torch.long)
                    self.train_step(x, y)

    @staticmethod
    def state_to_vector(state: str) -> torch.Tensor:
        return torch.tensor([int(char) for char in state], dtype=torch.float32)
