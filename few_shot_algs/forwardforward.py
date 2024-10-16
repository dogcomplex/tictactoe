import torch
import torch.nn as nn
import torch.optim as optim
from few_shot_algs.few_shot_alg import Algorithm

class ForwardForwardAlgorithm(Algorithm, nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=5, num_layers=3, learning_rate=0.01, threshold=2.0):
        Algorithm.__init__(self)
        nn.Module.__init__(self)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.threshold = threshold
        
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward_pass(self, x):
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
    
    def train_step(self, pos_x, neg_x):
        self.optimizer.zero_grad()
        
        pos_goodness = self.goodness(pos_x)
        neg_goodness = self.goodness(neg_x)
        
        loss = torch.mean(torch.relu(self.threshold - pos_goodness) + 
                          torch.relu(neg_goodness - self.threshold))
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, observation: str) -> int:
        # Convert observation string to tensor
        x = torch.tensor([int(digit) for digit in observation], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = self.forward_pass(x)
            prediction = torch.argmax(output, dim=1).item()
        
        return prediction
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        
        # Convert observation string to tensor
        x = torch.tensor([int(digit) for digit in observation], dtype=torch.float32).unsqueeze(0)
        
        # Create positive and negative samples
        pos_x = x.clone()
        neg_x = x.clone()
        
        # Modify one digit in neg_x to create a negative sample
        random_index = torch.randint(0, self.input_dim, (1,))
        neg_x[0, random_index] = (neg_x[0, random_index] + 1) % 3
        
        # Train the model
        loss = self.train_step(pos_x, neg_x)
