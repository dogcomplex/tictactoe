# Add this import at the top of your script
import random
from typing import List
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from few_shot_algs.few_shot_alg import Algorithm

# RandomForestAlgorithm
class RandomForestAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        )
        self.is_trained = False

    def predict(self, observation: str) -> int:
        # Convert observation to features
        features = self.observation_to_features(observation)

        if not self.is_trained or len(self.history) < 10:
            # Before training, randomly guess
            return np.random.randint(0, 5)
        else:
            # Predict using the trained model
            prediction = self.model.predict([features])
            return int(prediction[0])

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        # Retrain the model with the updated history
        self.train_model()

    def train_model(self):
        if len(self.history) >= 10:
            X = np.array([self.observation_to_features(obs) for obs, _, _ in self.history])
            y = np.array([label for _, _, label in self.history])
            self.model.fit(X, y)
            self.is_trained = True

    @staticmethod
    def observation_to_features(observation: str) -> np.ndarray:
        # Convert the base-3 observation string into a list of integers
        return np.array([int(char) for char in observation])
