# Add these imports at the top of your script
import torch
import torch.nn as nn
import torch.nn.functional as F
from few_shot_algs.few_shot_alg import Algorithm
from typing import List
import random

# PrototypicalNetworkAlgorithm
class PrototypicalNetworkAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 32  # Reduced from 64
        self.model = nn.Sequential(
            nn.Linear(9, 16),
            nn.ReLU(),
            nn.Linear(16, self.embedding_dim)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # Reduced learning rate
        self.is_trained = False
        self.device = torch.device('cpu')
        self.label_encoder = {}  # New: to map original labels to 0-based index

    def predict(self, observation: str) -> int:
        features = self.observation_to_features(observation)
        features = torch.tensor(features, dtype=torch.float32).to(self.device)

        if not self.is_trained:
            return random.randint(0, 4)
        else:
            embedding = self.model(features)
            # Compute distances to class prototypes
            distances = []
            for label in self.class_prototypes:
                prototype = self.class_prototypes[label]
                dist = F.pairwise_distance(embedding.unsqueeze(0), prototype.unsqueeze(0))
                distances.append(dist.item())
            # Predict the class with the nearest prototype
            prediction = int(min(self.class_prototypes.keys(), key=lambda x: distances[x]))
            return prediction

    def update_history(self, observation: str, guess: int, correct_label: int):
        # Encode the correct_label to ensure it starts from 0
        if correct_label not in self.label_encoder:
            self.label_encoder[correct_label] = len(self.label_encoder)
        encoded_label = self.label_encoder[correct_label]
        
        super().update_history(observation, guess, encoded_label)
        self.train_model()

    def train_model(self):
        if len(self.history) >= 10:  # Increased from 5
            self.is_trained = True
            X = []
            y = []
            for obs, _, label in self.history:
                features = self.observation_to_features(obs)
                X.append(features)
                y.append(label)  # This is now using the encoded label
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            y = torch.tensor(y, dtype=torch.long).to(self.device)

            # Train the model
            self.model.train()
            self.optimizer.zero_grad()
            embeddings = self.model(X)
            loss = self.prototypical_loss(embeddings, y)
            loss.backward()
            self.optimizer.step()

            # Update class prototypes
            self.update_class_prototypes(embeddings, y)

    def prototypical_loss(self, embeddings, labels):
        # Compute prototypes
        prototypes = []
        for label in torch.unique(labels):
            prototype = embeddings[labels == label].mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)

        # Compute distances and loss
        dists = torch.cdist(embeddings, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)
        loss = F.nll_loss(log_p_y, labels)
        return loss

    def update_class_prototypes(self, embeddings, labels):
        self.class_prototypes = {}
        for label in torch.unique(labels):
            prototype = embeddings[labels == label].mean(dim=0)
            self.class_prototypes[int(label.item())] = prototype.detach()

    @staticmethod
    def observation_to_features(observation: str) -> List[float]:
        return [float(int(char)) / 2 for char in observation]  # Normalize to [0, 1]
