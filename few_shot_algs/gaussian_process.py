# Add these imports at the top of your script
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from few_shot_algs.few_shot_alg import Algorithm
import numpy as np

# GaussianProcessAlgorithm
class GaussianProcessAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        # Adjust the kernel for the specific input space
        kernel = C(1.0) * RBF(length_scale=[1.0] * 9, length_scale_bounds=(1e-3, 1e5))
        self.model = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=9, random_state=42, max_iter_predict=100)
        self.is_trained = False

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
        if len(self.history) >= 5:  # Minimum samples for training
            X = np.array([self.observation_to_features(obs) for obs, _, _ in self.history])
            y = np.array([label for _, _, label in self.history])
            
            if len(np.unique(y)) >= 2:
                import warnings
                from sklearn.exceptions import ConvergenceWarning
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    self.model.fit(X, y)
                self.is_trained = True

    @staticmethod
    def observation_to_features(observation: str) -> np.ndarray:
        # Convert base-3 string to array of integers
        return np.array([int(char) for char in observation])
