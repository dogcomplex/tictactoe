import random
import time
import signal
from functools import partial
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
from few_shot_algs.forwardforward import ForwardForwardAlgorithm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import threading
from cycler import cycler

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
class TimeoutException(Exception):
    pass

class Tester:
    def __init__(self, problem_setup: ProblemSetupTicTacToe, algorithms: List[Algorithm], rounds: int, timeout: float = 2.0):
        self.problem_setup = problem_setup
        self.algorithms = algorithms
        self.rounds = rounds
        self.results = {alg.__class__.__name__: [] for alg in algorithms}
        self.cumulative_accuracy = {alg.__class__.__name__: [] for alg in algorithms}
        self.time_taken = {alg.__class__.__name__: 0 for alg in algorithms}
        self.cumulative_time = {alg.__class__.__name__: [] for alg in algorithms}
        self.timeout = timeout
        self.disqualified = {alg.__class__.__name__: False for alg in algorithms}

    def run_with_timeout(self, func, args):
        result = [TimeoutException()]
        def target():
            result[0] = func(*args)
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            raise TimeoutException()
        return result[0]

    def run_tests(self):
        for round_num in range(self.rounds):
            observation, correct_label = self.problem_setup.get_random_observation()
            for alg in self.algorithms:
                alg_name = alg.__class__.__name__
                if self.disqualified[alg_name]:
                    self.results[alg_name].append(None)
                    self.cumulative_accuracy[alg_name].append(self.cumulative_accuracy[alg_name][-1] if self.cumulative_accuracy[alg_name] else 0)
                    self.cumulative_time[alg_name].append(self.cumulative_time[alg_name][-1] if self.cumulative_time[alg_name] else self.timeout)
                    continue

                try:
                    start_time = time.time()
                    guess = self.run_with_timeout(alg.predict, (observation,))
                    self.run_with_timeout(alg.update_history, (observation, guess, correct_label))
                    end_time = time.time()

                    time_taken = end_time - start_time
                    self.time_taken[alg_name] += time_taken

                    is_correct = int(guess == correct_label)
                    self.results[alg_name].append(is_correct)

                    accuracy_so_far = sum(filter(None, self.results[alg_name])) / (round_num + 1)
                    avg_time_so_far = self.time_taken[alg_name] / (round_num + 1)

                    self.cumulative_accuracy[alg_name].append(accuracy_so_far)
                    self.cumulative_time[alg_name].append(avg_time_so_far)

                    print(f"Round {round_num+1}, Algorithm {alg_name}: "
                          f"Observation={observation}, Guess={guess}, Correct={correct_label}, "
                          f"Result={'Correct' if is_correct else 'Incorrect'}, "
                          f"Time={time_taken:.6f}s, "
                          f"Accuracy so far={accuracy_so_far:.2%}, "
                          f"Avg time so far={avg_time_so_far:.6f}s")

                except TimeoutException:
                    print(f"Round {round_num+1}, Algorithm {alg_name}: DISQUALIFIED (exceeded {self.timeout}s timeout)")
                    self.disqualified[alg_name] = True
                    self.results[alg_name].append(None)
                    self.cumulative_accuracy[alg_name].append(self.cumulative_accuracy[alg_name][-1] if self.cumulative_accuracy[alg_name] else 0)
                    self.cumulative_time[alg_name].append(self.timeout)

    def compute_metrics(self):
        for alg_name, results in self.results.items():
            if self.disqualified[alg_name]:
                print(f"Algorithm {alg_name}: DISQUALIFIED")
                valid_results = [r for r in results if r is not None]
                accuracy = sum(valid_results) / len(valid_results) if valid_results else 0
                avg_time = self.time_taken[alg_name] / len(valid_results) if valid_results else self.timeout
                print(f"  Accuracy before disqualification: {accuracy * 100:.2f}%")
                print(f"  Average time before disqualification: {avg_time:.6f} seconds")
            else:
                accuracy = sum(results) / len(results)
                avg_time = self.time_taken[alg_name] / len(results)
                print(f"Algorithm {alg_name}:")
                print(f"  Accuracy: {accuracy * 100:.2f}%")
                print(f"  Average time per prediction: {avg_time:.6f} seconds")

    def plot_results(self):
        # Set up a distinctive style cycle
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        plt.rc('axes', prop_cycle=(cycler(color=colors) +
                                   cycler(linestyle=line_styles*3) +
                                   cycler(marker=markers)))

        self._plot_accuracy()
        self._plot_compute_time()
        self._plot_accuracy_time_ratio()

    def _plot_accuracy(self):
        plt.figure(figsize=(12, 6))
        for alg_name, accuracies in self.cumulative_accuracy.items():
            plt.plot(range(1, len(accuracies) + 1), accuracies, label=alg_name, linewidth=2, markersize=4)
        
        plt.xlabel('Rounds')
        plt.ylabel('Cumulative Accuracy')
        plt.title('Algorithm Accuracy Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('algorithm_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_compute_time(self):
        plt.figure(figsize=(12, 6))
        for alg_name, times in self.cumulative_time.items():
            plt.plot(range(1, len(times) + 1), times, label=alg_name, linewidth=2, markersize=4)
        
        plt.xlabel('Rounds')
        plt.ylabel('Average Compute Time (seconds)')
        plt.title('Algorithm Compute Time Over Rounds')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('algorithm_compute_time.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_accuracy_time_ratio(self):
        plt.figure(figsize=(12, 6))
        for alg_name in self.cumulative_accuracy.keys():
            accuracies = np.array(self.cumulative_accuracy[alg_name])
            times = np.array(self.cumulative_time[alg_name])
            ratios = accuracies / times
            plt.plot(range(1, len(ratios) + 1), ratios, label=alg_name, linewidth=2, markersize=4)
        
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy / Compute Time Ratio')
        plt.title('Algorithm Efficiency (Accuracy/Time) Over Rounds')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('algorithm_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()

# Example Algorithm Implementation (to be expanded later)
class RandomGuessAlgorithm(Algorithm):
    def predict(self, observation: str) -> int:
        # Randomly guess a label
        return random.randint(0, 4)


class GaussianProcessAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        kernel = C(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e4))
        self.model = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=5)
        self.X = []
        self.y = []
        self.is_fitted = False

    def predict(self, observation: str) -> int:
        if self.is_fitted:
            return self.model.predict([self.state_to_vector(observation)])[0]
        else:
            return random.randint(0, 4)

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        self.X.append(self.state_to_vector(observation))
        self.y.append(correct_label)
        if len(self.X) > 1 and len(set(self.y)) > 1:
            self.model.fit(self.X, self.y)
            self.is_fitted = True

    @staticmethod
    def state_to_vector(state: str) -> List[int]:
        return [int(char) for char in state]

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
        #GPT2Algorithm()
        ForwardForwardAlgorithm()
    ]
    tester = Tester(problem_setup, algorithms, rounds=300)
    tester.run_tests()
    tester.compute_metrics()
    tester.plot_results()
    print("Results graphs saved as 'algorithm_accuracy.png', 'algorithm_compute_time.png', and 'algorithm_efficiency.png'")
