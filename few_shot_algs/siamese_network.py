# Add these imports at the top of your script
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from few_shot_algs.few_shot_alg import Algorithm
from typing import List

# SiameseNetworkAlgorithm
class SiameseNetworkAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.input_dim = 9 * 3  # 9 digits, each represented by 3 bits (one-hot)
        self.embedding_dim = 32  # Reduced from 64
        self.model = EmbeddingNetwork(self.input_dim, self.embedding_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # Reduced learning rate
        self.criterion = nn.BCEWithLogitsLoss()
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def predict(self, observation: str) -> int:
        features = self.observation_to_features(observation)
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        features = features.unsqueeze(0)

        if not self.is_trained:
            return random.randint(0, 4)
        else:
            min_distance = float('inf')
            predicted_label = random.randint(0, 4)
            # Compare with stored embeddings
            for obs, _, label in self.history:
                stored_features = self.observation_to_features(obs)
                stored_features = torch.tensor(stored_features, dtype=torch.float32).to(self.device)
                stored_features = stored_features.unsqueeze(0)
                with torch.no_grad():
                    emb1 = self.model(features)
                    emb2 = self.model(stored_features)
                    distance = F.pairwise_distance(emb1, emb2).item()
                if distance < min_distance:
                    min_distance = distance
                    predicted_label = label
            return predicted_label

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        self.train_model()

    def train_model(self):
        if len(self.history) >= 5:
            self.is_trained = True
            dataset = SiameseDataset(self.history, self.observation_to_features)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Increased batch size
            self.model.train()
            for epoch in range(5):  # Increased number of epochs
                for batch in dataloader:
                    (x1, x2), labels = batch
                    x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    output1 = self.model(x1)
                    output2 = self.model(x2)
                    distances = F.pairwise_distance(output1, output2)
                    loss = self.criterion(-distances, labels.float())
                    loss.backward()
                    self.optimizer.step()

    @staticmethod
    def observation_to_features(observation: str) -> List[float]:
        features = []
        for char in observation:
            if char == '0':
                features.extend([1, 0, 0])
            elif char == '1':
                features.extend([0, 1, 0])
            else:  # '2'
                features.extend([0, 0, 1])
        return features

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(EmbeddingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SiameseDataset(Dataset):
    def __init__(self, history, observation_to_features):
        self.pairs = []
        self.labels = []
        self.observation_to_features = observation_to_features
        labels = [label for _, _, label in history]
        unique_labels = set(labels)
        label_to_indices = {label: [i for i, (_, _, l) in enumerate(history) if l == label] for label in unique_labels}

        for i in range(len(history)):
            obs1, _, label1 = history[i]
            # Positive pair
            pos_indices = label_to_indices[label1]
            if len(pos_indices) > 1:  # Ensure we have at least two samples of this label
                pos_index = random.choice([idx for idx in pos_indices if idx != i])
                obs2, _, _ = history[pos_index]
                self.pairs.append((self.observation_to_features(obs1), self.observation_to_features(obs2)))
                self.labels.append(1)

            # Negative pair
            if len(unique_labels) > 1:  # Ensure we have at least two unique labels
                neg_label = random.choice(list(unique_labels - {label1}))
                neg_index = random.choice(label_to_indices[neg_label])
                obs2, _, _ = history[neg_index]
                self.pairs.append((self.observation_to_features(obs1), self.observation_to_features(obs2)))
                self.labels.append(0)

    def __getitem__(self, index):
        x1, x2 = self.pairs[index]
        label = self.labels[index]
        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return (x1, x2), label

    def __len__(self):
        return len(self.pairs)
