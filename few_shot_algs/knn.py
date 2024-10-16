# Add this import at the top of your script
from sklearn.neighbors import KNeighborsClassifier
from few_shot_algs.few_shot_alg import Algorithm
from typing import List
import numpy as np

# KNNAlgorithm
class KNNAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        # Adjust n_neighbors based on the label space size
        self.model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='hamming')
        self.is_trained = False

    def predict(self, observation: str) -> int:
        features = self.observation_to_features(observation)

        if not self.is_trained or len(self.history) < 5:
            # Before training or with insufficient data, use a weighted random guess
            return self.weighted_random_guess()
        else:
            prediction = self.model.predict([features])
            return int(prediction[0])

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        # Retrain the model with the updated history
        self.train_model()

    def train_model(self):
        if len(self.history) >= 5:  # Train only if we have at least 5 samples
            X = []
            y = []
            for obs, _, label in self.history:
                features = self.observation_to_features(obs)
                X.append(features)
                y.append(label)
            self.model.fit(X, y)
            self.is_trained = True

    @staticmethod
    def observation_to_features(observation: str) -> List[int]:
        # Convert base-3 string to list of integers
        return [int(char) for char in observation]

    def weighted_random_guess(self) -> int:
        if not self.history:
            return np.random.randint(0, 5)
        
        label_counts = np.zeros(5)
        for _, _, label in self.history:
            label_counts[label] += 1
        
        # Add a small constant to avoid zero probabilities
        label_counts += 0.1
        
        # Normalize to get probabilities
        probs = label_counts / np.sum(label_counts)
        
        return np.random.choice(5, p=probs)
