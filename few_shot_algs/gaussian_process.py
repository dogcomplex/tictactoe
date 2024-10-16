# Add these imports at the top of your script
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from few_shot_algs.few_shot_alg import Algorithm
from typing import List
import numpy as np
import random
from collections import Counter

# GaussianProcessAlgorithm
class GaussianProcessAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        # Adjust the kernel for the specific input space
        kernel = C(1.0) * RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e2))
        self.model = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        self.is_trained = False
        self.most_common_class = None

    def predict(self, observation: str) -> int:
        features = self.observation_to_features(observation)
        if not self.is_trained:
            return np.random.randint(0, 5)  # Random guess from 0 to 4
        else:
            prediction = self.model.predict([features])
            return int(prediction[0])

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        self.train_model()

    def train_model(self):
        if len(self.history) >= 10:  # Increased minimum samples for training
            X = []
            y = []
            for obs, _, label in self.history:
                features = self.observation_to_features(obs)
                X.append(features)
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            # Check if we have at least two distinct classes
            if len(np.unique(y)) >= 2:
                self.model.fit(X, y)
                self.is_trained = True
            else:
                # If we don't have enough distinct classes, use the most common class
                self.most_common_class = Counter(y).most_common(1)[0][0]
                self.is_trained = False

    @staticmethod
    def observation_to_features(observation: str) -> List[float]:
        # Normalize features to be between 0 and 1
        return [float(char) / 2 for char in observation]
