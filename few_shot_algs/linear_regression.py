from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from few_shot_algs.few_shot_alg import Algorithm
import numpy as np
from typing import List
import random

class LinearRegressionAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        try:
            self.encoder = OneHotEncoder(sparse_output=False, categories=[['0', '1', '2']] * 9)
        except TypeError:
            self.encoder = OneHotEncoder(sparse=False, categories=[['0', '1', '2']] * 9)
        self.is_trained = False
        self.encoder_fitted = False

    def predict(self, observation: str) -> int:
        features = self.observation_to_features(observation)
        if not self.is_trained:
            return np.random.randint(0, 5)
        else:
            prediction = self.model.predict(features)
            return int(np.clip(np.round(prediction[0]), 0, 4))

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        self.train_model()

    def train_model(self):
        if len(self.history) >= 5:
            X = np.vstack([self.observation_to_features(obs) for obs, _, _ in self.history])
            y = np.array([label for _, _, label in self.history])
            self.model.fit(X, y)
            self.is_trained = True

    def observation_to_features(self, observation: str) -> np.ndarray:
        obs_array = np.array(list(observation)).reshape(1, -1)
        if not self.encoder_fitted:
            self.encoder.fit(obs_array)
            self.encoder_fitted = True
        return self.encoder.transform(obs_array)
