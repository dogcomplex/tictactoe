import random
from collections import Counter
from few_shot_algs.few_shot_alg import Algorithm

class DistributionApproximatorAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.label_counts = Counter()
        self.total_samples = 0
        self.smoothing_factor = 0.1  # Laplace smoothing factor

    def predict(self, observation: str) -> int:
        if self.total_samples == 0:
            return random.randint(0, 4)
        
        # Calculate probabilities with Laplace smoothing
        probabilities = []
        for label in range(5):
            count = self.label_counts[label]
            probability = (count + self.smoothing_factor) / (self.total_samples + 5 * self.smoothing_factor)
            probabilities.append(probability)
        
        # Randomly select a label based on the approximated distribution
        return random.choices(range(5), weights=probabilities)[0]

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        self.label_counts[correct_label] += 1
        self.total_samples += 1

    def get_distribution(self):
        if self.total_samples == 0:
            return [0.2] * 5  # Uniform distribution if no samples
        
        distribution = []
        for label in range(5):
            count = self.label_counts[label]
            probability = (count + self.smoothing_factor) / (self.total_samples + 5 * self.smoothing_factor)
            distribution.append(probability)
        
        return distribution

