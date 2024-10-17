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
        
        # Adjust the input dimension of the first layer to account for the one-hot encoded class
        self.layers = nn.ModuleList([
            nn.Linear(input_dim + output_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Add separate optimizers for each layer
        self.optimizers = [optim.Adam(layer.parameters(), lr=learning_rate) for layer in self.layers]
    
    def goodness(self, x):
        total_goodness = 0
        for layer in self.layers:
            x = layer(x)
            total_goodness += torch.sum(x ** 2, dim=1)
            x = torch.relu(x)
        return total_goodness
    
    def train_step(self, x_pos, x_neg):
        for layer, optimizer in zip(self.layers, self.optimizers):
            optimizer.zero_grad()
            
            # Compute goodness for positive and negative samples
            g_pos = torch.sum(torch.relu(layer(x_pos)) ** 2, dim=1)
            g_neg = torch.sum(torch.relu(layer(x_neg)) ** 2, dim=1)
            
            # Compute loss
            loss = torch.mean(torch.log(1 + torch.exp(self.threshold - g_pos))) + \
                   torch.mean(torch.log(1 + torch.exp(g_neg - self.threshold)))
            
            # Backward and optimize for this layer only
            loss.backward()
            optimizer.step()
            
            # Detach x_pos and x_neg for the next layer
            x_pos = layer(x_pos).detach()
            x_neg = layer(x_neg).detach()

    def predict(self, observation: str) -> int:
        x = self.state_to_vector(observation).unsqueeze(0)
        
        with torch.no_grad():
            goodness_per_class = []
            for class_idx in range(self.output_dim):
                x_with_class = torch.cat([x, torch.eye(self.output_dim)[class_idx].unsqueeze(0)], dim=1)
                goodness = self.goodness(x_with_class)
                goodness_per_class.append(goodness.item())
            
            prediction = goodness_per_class.index(max(goodness_per_class))
        
        return prediction
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        
        # Train on the entire history for multiple epochs
        if len(self.history) >= self.batch_size:
            for _ in range(self.epochs):
                random.shuffle(self.history)
                for i in range(0, len(self.history), self.batch_size):
                    batch = self.history[i:i+self.batch_size]
                    x_pos = torch.tensor([
                        [int(digit) for digit in obs] + [0] * self.output_dim 
                        for obs, _, label in batch
                    ], dtype=torch.float32)
                    for j, (_, _, label) in enumerate(batch):
                        x_pos[j, self.input_dim + label] = 1
                    
                    # Generate negative samples
                    x_neg = x_pos.clone()
                    for j in range(len(x_neg)):
                        wrong_label = random.choice([l for l in range(self.output_dim) if l != batch[j][2]])
                        x_neg[j, self.input_dim:] = 0
                        x_neg[j, self.input_dim + wrong_label] = 1
                    
                    self.train_step(x_pos, x_neg)

    @staticmethod
    def state_to_vector(state: str) -> torch.Tensor:
        return torch.tensor([int(char) for char in state], dtype=torch.float32)
