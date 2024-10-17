from few_shot_algs.few_shot_alg import Algorithm
from recipes import TicTacToeAlgorithm, map_observation_to_tensor, unpack_bits, process_observations
import random
import torch

class LocusBanditAlgorithm(Algorithm):
    def __init__(self, epsilon=0.1, learning_rate=0.1):
        super().__init__()
        self.ttt_algorithm = TicTacToeAlgorithm(use_disk_cache=False)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.hypothesis_values = torch.full((self.ttt_algorithm.hypothesis_manager.total_hypotheses,), 0.5, device=self.ttt_algorithm.hypothesis_manager.device)
        self.device = self.ttt_algorithm.hypothesis_manager.device
        self.observations = []
        self.output_labels = ['C', 'W', 'L', 'D', 'E']

    def predict(self, observation: str) -> int:
        obs_tensor = map_observation_to_tensor(observation, None)
        remaining_hypotheses = self.ttt_algorithm.hypothesis_manager.get_matching_hypotheses(obs_tensor)
        
        print(f"\nPredicting for observation: {observation}")
        print(f"Remaining hypotheses: {len(remaining_hypotheses)}")

        if len(remaining_hypotheses) == 0:
            print("No matching hypotheses. Making random guess.")
            return random.randint(0, 4)

        matching_hypotheses = unpack_bits(self.ttt_algorithm.hypothesis_manager.hypotheses[remaining_hypotheses])
        
        output_counts = matching_hypotheses[:, 27:].sum(dim=0)
        total_matches = len(remaining_hypotheses)

        unweighted_probs = output_counts.float() / total_matches
        print("\nUnweighted probabilities:")
        self._print_probabilities(unweighted_probs)

        if len(self.observations) > 0:
            observation_tensors = process_observations(self.observations)
            stats = self.ttt_algorithm.hypothesis_manager.calculate_hypothesis_stats(matching_hypotheses, observation_tensors)
            weights = torch.tensor([max(1, valid) for _, valid, _ in stats], device=self.device)
            weighted_output_counts = (matching_hypotheses[:, 27:].float() * weights.unsqueeze(1)).sum(dim=0)
            weighted_probs = weighted_output_counts / weights.sum()
            print("\nWeighted probabilities:")
            self._print_probabilities(weighted_probs)
        else:
            weighted_probs = unweighted_probs
            print("No previous observations. Using unweighted probabilities.")

        combined_probs = weighted_probs * self.hypothesis_values[remaining_hypotheses].mean()
        print("\nCombined probabilities (weighted * bandit values):")
        self._print_probabilities(combined_probs)

        if random.random() < self.epsilon:
            prediction = torch.multinomial(combined_probs, 1).item()
            print(f"\nExploring: randomly selected prediction based on combined probabilities.")
        else:
            prediction = combined_probs.argmax().item()
            print(f"\nExploiting: selected prediction with highest combined probability.")

        print(f"Final prediction: {self.output_labels[prediction]}")
        return prediction

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        correct_label_str = self.output_labels[correct_label]
        
        if isinstance(guess, Exception):
            guess_str = 'E'
        else:
            guess_str = self.output_labels[guess]
        
        print(f"\nUpdating history:")
        print(f"Observation: {observation}")
        print(f"Guess: {guess_str}")
        print(f"Correct label: {correct_label_str}")

        self.ttt_algorithm.update_history(observation, guess_str, correct_label_str)
        self.observations.append((observation, correct_label_str))

        obs_tensor = map_observation_to_tensor(observation, correct_label_str)
        remaining_hypotheses = self.ttt_algorithm.hypothesis_manager.get_matching_hypotheses(obs_tensor)

        print(f"Matching hypotheses after update: {len(remaining_hypotheses)}")

        if len(remaining_hypotheses) > 0:
            hypotheses_batch = unpack_bits(self.ttt_algorithm.hypothesis_manager.hypotheses[remaining_hypotheses])
            hypothesis_predictions = hypotheses_batch[:, 27:].int().argmax(dim=1)
            
            rewards = (hypothesis_predictions == correct_label).float()
            self.hypothesis_values[remaining_hypotheses] += self.learning_rate * (rewards - self.hypothesis_values[remaining_hypotheses])

            print("\nHypothesis value updates:")
            print(f"Mean reward: {rewards.mean().item():.4f}")
            print(f"Mean hypothesis value before update: {self.hypothesis_values[remaining_hypotheses].mean().item():.4f}")
            print(f"Mean hypothesis value after update: {self.hypothesis_values[remaining_hypotheses].mean().item():.4f}")

    def _print_probabilities(self, probs):
        for label, prob in zip(self.output_labels, probs):
            print(f"{label}: {prob.item():.4f}")

    def map_observation_to_tensor(self, observation: str, output_char: str = None) -> torch.Tensor:
        return map_observation_to_tensor(observation, output_char)
