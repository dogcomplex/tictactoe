import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from few_shot_algs.few_shot_alg import Algorithm
from concurrent.futures import TimeoutError as TimeoutException

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAlgorithm(Algorithm):
    def __init__(self, input_dim=9, output_dim=5, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=1000, batch_size=32):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(input_dim, output_dim).to(self.device)
        self.target_dqn = DQN(input_dim, output_dim).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.output_dim = output_dim

    def predict(self, observation: str) -> int:
        state = self.state_to_tensor(observation).to(self.device)
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        with torch.no_grad():
            q_values = self.dqn(state)
            return torch.argmax(q_values).item()

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        
        # Check if guess is a TimeoutException
        if isinstance(guess, TimeoutException):
            return  # Skip training if there was a timeout
        
        state = self.state_to_tensor(observation).to(self.device)
        reward = 1 if guess == correct_label else -1
        next_state = state  # In this case, next_state is the same as the current state
        
        self.memory.append((state, guess, reward, next_state, correct_label))
        
        if len(self.memory) >= self.batch_size:
            self.train_model()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_model(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, labels = zip(*batch)

        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)

        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_dqn(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values)

        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if len(self.memory) % 100 == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

    @staticmethod
    def state_to_tensor(state: str) -> torch.Tensor:
        return torch.tensor([int(char) for char in state], dtype=torch.float32).unsqueeze(0)
