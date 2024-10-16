import random
import time
from typing import List, Tuple, Dict
from few_shot_algs.few_shot_alg import Algorithm
from few_shot_algs.random_forest import RandomForestAlgorithm
from few_shot_algs.knn import KNNAlgorithm
from few_shot_algs.prototypical_network import PrototypicalNetworkAlgorithm
from few_shot_algs.bayesian_nn import BayesianNNAlgorithm
from few_shot_algs.linear_regression import LinearRegressionAlgorithm
from few_shot_algs.siamese_network import SiameseNetworkAlgorithm
from few_shot_algs.gaussian_process import GaussianProcessAlgorithm
from few_shot_algs.transformer import TransformerAlgorithm
from few_shot_algs.gpt2 import GPT2Algorithm
import matplotlib.pyplot as plt
import numpy as np

from tictactoe import attempt_solve, label_space

# 1. Problem Setup
class ProblemSetupRandom:
    def __init__(self):
        self.state_space_size = 3 ** 9  # 9 digits in base 3
        self.label_space_size = 5       # Labels in base 5
        self.state_to_label = self.generate_state_label_mapping()
    
    def generate_state_label_mapping(self) -> Dict[str, int]:
        # For simplicity, create a random mapping
        mapping = {}
        for i in range(self.state_space_size):
            state = self.int_to_base(i, 3, 9)
            label = random.randint(0, self.label_space_size - 1)
            mapping[state] = label
        return mapping

    @staticmethod
    def int_to_base(number: int, base: int, length: int) -> str:
        digits = []
        for _ in range(length):
            digits.append(str(number % base))
            number //= base
        return ''.join(reversed(digits))

    def get_random_observation(self) -> Tuple[str, int]:
        state = random.choice(list(self.state_to_label.keys()))
        label = self.state_to_label[state]
        return state, label



class ProblemSetupTicTacToe(ProblemSetupRandom):
    def __init__(self):
        super().__init__()
        self.state_space_size = 3 ** 9  # 9 digits in base 3
        self.label_space_size = 5       # Labels in base 5
        self.state_to_label = self.generate_state_label_mapping()

    def get_random_observation(self) -> Tuple[str, int]:
        results, all_valid = attempt_solve(1)
        print(results)
        return results[0]['board'], label_space.index(results[0]['solver_result'])


# 3. Tester Class
class Tester:
    def __init__(self, problem_setup: ProblemSetupTicTacToe, algorithms: List[Algorithm], rounds: int):
        self.problem_setup = problem_setup
        self.algorithms = algorithms
        self.rounds = rounds
        self.results = {alg.__class__.__name__: [] for alg in algorithms}
        self.cumulative_accuracy = {alg.__class__.__name__: [] for alg in algorithms}
        self.time_taken = {alg.__class__.__name__: 0 for alg in algorithms}
        self.cumulative_time = {alg.__class__.__name__: [] for alg in algorithms}

    def run_tests(self):
        for round_num in range(self.rounds):
            observation, correct_label = self.problem_setup.get_random_observation()
            for alg in self.algorithms:
                start_time = time.time()
                guess = alg.predict(observation)
                end_time = time.time()
                
                time_taken = end_time - start_time
                self.time_taken[alg.__class__.__name__] += time_taken
                
                alg.update_history(observation, guess, correct_label)
                is_correct = int(guess == correct_label)
                self.results[alg.__class__.__name__].append(is_correct)
                
                accuracy_so_far = sum(self.results[alg.__class__.__name__]) / (round_num + 1)
                avg_time_so_far = self.time_taken[alg.__class__.__name__] / (round_num + 1)
                
                self.cumulative_accuracy[alg.__class__.__name__].append(accuracy_so_far)
                self.cumulative_time[alg.__class__.__name__].append(avg_time_so_far)
                
                print(f"Round {round_num+1}, Algorithm {alg.__class__.__name__}: "
                      f"Observation={observation}, Guess={guess}, Correct={correct_label}, "
                      f"Result={'Correct' if is_correct else 'Incorrect'}, "
                      f"Time={time_taken:.6f}s, "
                      f"Accuracy so far={accuracy_so_far:.2%}, "
                      f"Avg time so far={avg_time_so_far:.6f}s")

    def compute_metrics(self):
        for alg_name, results in self.results.items():
            accuracy = sum(results) / len(results)
            avg_time = self.time_taken[alg_name] / self.rounds
            print(f"Algorithm {alg_name}:")
            print(f"  Accuracy: {accuracy * 100:.2f}%")
            print(f"  Average time per prediction: {avg_time:.6f} seconds")

    def plot_results(self):
        self._plot_accuracy()
        self._plot_compute_time()
        self._plot_accuracy_time_ratio()

    def _plot_accuracy(self):
        plt.figure(figsize=(12, 6))
        for alg_name, accuracies in self.cumulative_accuracy.items():
            plt.plot(range(1, self.rounds + 1), accuracies, label=alg_name)
        
        plt.xlabel('Rounds')
        plt.ylabel('Cumulative Accuracy')
        plt.title('Algorithm Accuracy Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('algorithm_accuracy.png')
        plt.close()

    def _plot_compute_time(self):
        plt.figure(figsize=(12, 6))
        for alg_name, times in self.cumulative_time.items():
            plt.plot(range(1, self.rounds + 1), times, label=alg_name)
        
        plt.xlabel('Rounds')
        plt.ylabel('Average Compute Time (seconds)')
        plt.title('Algorithm Compute Time Over Rounds')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('algorithm_compute_time.png')
        plt.close()

    def _plot_accuracy_time_ratio(self):
        plt.figure(figsize=(12, 6))
        for alg_name in self.cumulative_accuracy.keys():
            accuracies = np.array(self.cumulative_accuracy[alg_name])
            times = np.array(self.cumulative_time[alg_name])
            ratios = accuracies / times
            plt.plot(range(1, self.rounds + 1), ratios, label=alg_name)
        
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy / Compute Time Ratio')
        plt.title('Algorithm Efficiency (Accuracy/Time) Over Rounds')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('algorithm_efficiency.png')
        plt.close()

# Example Algorithm Implementation (to be expanded later)
class RandomGuessAlgorithm(Algorithm):
    def predict(self, observation: str) -> int:
        # Randomly guess a label
        return random.randint(0, 4)


if __name__ == "__main__":
    problem_setup = ProblemSetupTicTacToe()
    algorithms = [
        RandomGuessAlgorithm(),
        RandomForestAlgorithm(),
        KNNAlgorithm(),
        PrototypicalNetworkAlgorithm(),
        BayesianNNAlgorithm(),
        LinearRegressionAlgorithm(),
        SiameseNetworkAlgorithm(),
        GaussianProcessAlgorithm(),
        TransformerAlgorithm(),
        GPT2Algorithm()
    ]
    tester = Tester(problem_setup, algorithms, rounds=300)
    tester.run_tests()
    tester.compute_metrics()
    tester.plot_results()
    print("Results graphs saved as 'algorithm_accuracy.png', 'algorithm_compute_time.png', and 'algorithm_efficiency.png'")
