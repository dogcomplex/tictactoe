# Add these imports at the top of your script
import torch
import torch.nn as nn
from few_shot_algs.few_shot_alg import Algorithm
from typing import List
import random

# TransformerAlgorithm
class TransformerAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 16  # Reduced from 32
        self.model = TransformerClassifier(self.embedding_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # Reduced learning rate
        self.criterion = nn.CrossEntropyLoss()
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def predict(self, observation: str) -> int:
        features = self.observation_to_features(observation)
        features = torch.tensor(features, dtype=torch.long).unsqueeze(0).to(self.device)

        if not self.is_trained:
            return random.randint(0, 4)
        else:
            with torch.no_grad():
                output = self.model(features)
                prediction = torch.argmax(output, dim=1)
            return int(prediction.item())

    def update_history(self, observation: str, guess: int, correct_label: int) -> None:
        super().update_history(observation, guess, correct_label)
        self.train_model()

    def train_model(self) -> None:
        if len(self.history) >= 3:  # Reduced from 5
            self.is_trained = True
            X = torch.tensor([self.observation_to_features(obs) for obs, _, _ in self.history[-10:]], dtype=torch.long).to(self.device)
            y = torch.tensor([label for _, _, label in self.history[-10:]], dtype=torch.long).to(self.device)

            for _ in range(5):  # Multiple training iterations
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

    @staticmethod
    def observation_to_features(observation: str) -> List[int]:
        # Since transformer uses embeddings, convert digits to integers
        return [int(char) for char in observation]

class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(3, embedding_dim)  # Correct: 3 for digits 0,1,2
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=2, dim_feedforward=32)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embedding_dim * 9, 5)  # Correct: 5 output labels
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
