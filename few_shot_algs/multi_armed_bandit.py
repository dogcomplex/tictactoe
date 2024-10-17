import random
from typing import Dict, List
from few_shot_algs.few_shot_alg import Algorithm

class MultiArmedBanditAlgorithm(Algorithm):
    def __init__(self, epsilon: float = 0.1, initial_value: float = 0.5):
        super().__init__()
        self.num_arms = 5  # 5 labels: 0/1/2/3/4
        self.epsilon = epsilon
        self.arm_values: Dict[int, float] = {i: initial_value for i in range(self.num_arms)}
        self.arm_counts: Dict[int, int] = {i: 0 for i in range(self.num_arms)}
        self.state_arm_values: Dict[str, Dict[int, float]] = {}

    def predict(self, observation: str) -> int:
        if observation not in self.state_arm_values:
            self.state_arm_values[observation] = {i: 0.5 for i in range(self.num_arms)}

        if random.random() < self.epsilon:
            # Explore: choose a random arm
            return random.randint(0, self.num_arms - 1)
        else:
            # Exploit: choose the arm with the highest value for this state
            return max(self.state_arm_values[observation], key=self.state_arm_values[observation].get)

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        
        # Update the chosen arm's value and count
        self.arm_counts[guess] += 1
        n = self.arm_counts[guess]
        value = self.arm_values[guess]
        
        # Update the overall arm's value using incremental average
        self.arm_values[guess] = ((n - 1) / n) * value + (1 / n) * (1 if guess == correct_label else 0)

        # Update the state-specific arm value
        state_value = self.state_arm_values[observation][guess]
        self.state_arm_values[observation][guess] = 0.9 * state_value + 0.1 * (1 if guess == correct_label else 0)

    @staticmethod
    def state_to_vector(state: str) -> List[int]:
        return [int(char) for char in state]

    def __str__(self):
        return f"MultiArmedBanditAlgorithm(epsilon={self.epsilon})"

