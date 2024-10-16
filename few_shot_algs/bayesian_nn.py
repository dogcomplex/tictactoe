# Add these imports at the top of your script
import tensorflow as tf
import numpy as np
from few_shot_algs.few_shot_alg import Algorithm
from typing import List
import random

# BayesianNeuralNetworkAlgorithm
class BayesianNNAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model = self.build_model()
        self.is_trained = False

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(9,)),
            tf.keras.layers.Dense(27, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(18, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        model.compile(optimizer=self.optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def predict(self, observation: str) -> int:
        features = self.observation_to_features(observation)
        features = np.array([features], dtype=np.float32)

        if not self.is_trained:
            return random.randint(0, 4)
        else:
            prediction = self.model.predict(features, verbose=0)
            return int(np.argmax(prediction[0]))

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        self.train_model()

    def train_model(self):
        if len(self.history) >= 3:  # Reduced minimum training samples
            self.is_trained = True
            X = []
            y = []
            for obs, _, label in self.history:
                features = self.observation_to_features(obs)
                X.append(features)
                y.append(label)
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int32)

            self.model.fit(X, y, epochs=5, batch_size=min(32, len(X)), verbose=0)

    @staticmethod
    def observation_to_features(observation: str) -> list:
        return [int(char) / 2 for char in observation]  # Normalize to [0, 1]
