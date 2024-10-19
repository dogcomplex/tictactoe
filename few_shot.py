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
import csv
from datetime import datetime
import os

from tictactoe import attempt_solve, label_space, random_board, all_answers
from few_shot_algs.dqn import DQNAlgorithm
from few_shot_algs.diffusion import DiffusionAlgorithm
from few_shot_algs.distribution_approximator import DistributionApproximatorAlgorithm
from few_shot_algs.multi_armed_bandit import MultiArmedBanditAlgorithm
from few_shot_algs.locus import LocusAlgorithm
from few_shot_algs.locus_bandit import LocusBanditAlgorithm
from few_shot_algs.reptile import MAMLReptileAlgorithm
from few_shot_algs.sat import SATAlgorithm

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
        board = random_board()
        answer = all_answers[board]
        answer_int = label_space.index(answer)
        print(board, answer_int)
        return board, answer_int


# 3. Tester Class
class TimeoutException(Exception):
    pass

class Tester:
    def __init__(self, problem_setup: ProblemSetupTicTacToe, algorithms: List[Algorithm], rounds: int, timeout: float = 200.0, ON_THE_HOUSE: bool = True):
        self.problem_setup = problem_setup
        self.algorithms = algorithms
        self.rounds = rounds
        self.results = {alg.__class__.__name__: [] for alg in algorithms}
        self.cumulative_accuracy = {alg.__class__.__name__: [] for alg in algorithms}
        self.time_taken = {alg.__class__.__name__: 0 for alg in algorithms}
        self.cumulative_time = {alg.__class__.__name__: [] for alg in algorithms}
        self.timeout = timeout
        self.disqualified = {alg.__class__.__name__: False for alg in algorithms}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f"results/run_{self.timestamp}"
        self.log_filename = f"{self.results_dir}/algorithm_results.csv"
        
        # Create the results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        self.ON_THE_HOUSE = ON_THE_HOUSE
        self.observation_cache = {}

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
        with open(self.log_filename, 'w', newline='') as log_file:
            csv_writer = csv.writer(log_file)
            csv_writer.writerow(['Round', 'Algorithm', 'Observation', 'Guess', 'Correct Label', 'Is Correct', 'Time Taken', 'On The House'])

            for round_num in range(self.rounds):
                observation, correct_label = self.problem_setup.get_random_observation()
                on_the_house = False

                if self.ON_THE_HOUSE and observation in self.observation_cache:
                    on_the_house = True
                    correct_label = self.observation_cache[observation]
                else:
                    self.observation_cache[observation] = correct_label

                for alg in self.algorithms:
                    alg_name = alg.__class__.__name__
                    if self.disqualified[alg_name]:
                        csv_writer.writerow([round_num + 1, alg_name, observation, 'DISQUALIFIED', correct_label, False, self.timeout, on_the_house])
                        self.results[alg_name].append(None)
                        self.cumulative_accuracy[alg_name].append(self.cumulative_accuracy[alg_name][-1] if self.cumulative_accuracy[alg_name] else 0)
                        self.cumulative_time[alg_name].append(self.cumulative_time[alg_name][-1] if self.cumulative_time[alg_name] else self.timeout)
                        continue

                    try:
                        start_time = time.time()
                        guess = self.run_with_timeout(alg.predict, (observation,))
                        
                        if on_the_house:
                            guess = correct_label

                        self.run_with_timeout(alg.update_history, (observation, guess, correct_label))
                        end_time = time.time()

                        time_taken = end_time - start_time
                        self.time_taken[alg_name] += time_taken

                        is_correct = int(guess == correct_label)
                        self.results[alg_name].append(is_correct)

                        # Calculate accuracy based on last 100 results
                        last_100_results = self.results[alg_name][-100:]
                        accuracy_so_far = sum(last_100_results) / len(last_100_results)
                        
                        avg_time_so_far = self.time_taken[alg_name] / (round_num + 1)

                        self.cumulative_accuracy[alg_name].append(accuracy_so_far)
                        self.cumulative_time[alg_name].append(avg_time_so_far)

                        # Log the result
                        csv_writer.writerow([round_num + 1, alg_name, observation, guess, correct_label, is_correct, time_taken, on_the_house])

                        print(f"Round {round_num+1}, Algorithm {alg_name}: "
                              f"Observation={observation}, Guess={guess}, Correct={correct_label}, "
                              f"Result={'Correct' if is_correct else 'Incorrect'}, "
                              f"Time={time_taken:.6f}s, "
                              f"Accuracy (last 100)={accuracy_so_far:.2%}, "
                              f"Avg time so far={avg_time_so_far:.6f}s, "
                              f"On The House: {'Yes' if on_the_house else 'No'}")

                    except TimeoutException:
                        # Log the result
                        csv_writer.writerow([round_num + 1, alg_name, observation, 'TIMEOUT', correct_label, False, self.timeout, on_the_house])
                        print(f"Round {round_num+1}, Algorithm {alg_name}: DISQUALIFIED (exceeded {self.timeout}s timeout)")
                        self.disqualified[alg_name] = True
                        self.results[alg_name].append(None)
                        self.cumulative_accuracy[alg_name].append(self.cumulative_accuracy[alg_name][-1] if self.cumulative_accuracy[alg_name] else 0)
                        self.cumulative_time[alg_name].append(self.timeout)

        print(f"Detailed results have been saved to {self.log_filename}")

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
        num_algorithms = len(self.algorithms)
        colors = plt.cm.tab20(np.linspace(0, 1, num_algorithms))
        line_styles = ['-', '--', '-.', ':'] * (num_algorithms // 4 + 1)
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'] * (num_algorithms // 10 + 1)
        
        plt.rc('axes', prop_cycle=(cycler(color=colors[:num_algorithms]) +
                                   cycler(linestyle=line_styles[:num_algorithms])))

        self._plot_accuracy()
        self._plot_compute_time()
        self._plot_accuracy_time_ratio()

    def _plot_accuracy(self):
        plt.figure(figsize=(10, 6))
        for algorithm in self.algorithms:
            alg_name = algorithm.__class__.__name__
            accuracies = self.cumulative_accuracy[alg_name]
            
            # Calculate marker interval based on number of rounds
            marker_interval = max(1, len(accuracies) // 20)
            
            plt.plot(range(1, len(accuracies) + 1), accuracies, 
                     label=alg_name, marker='o', markersize=4, markevery=marker_interval)
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Algorithm Accuracy over Rounds')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.results_dir}/algorithm_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_compute_time(self):
        plt.figure(figsize=(10, 6))
        for algorithm in self.algorithms:
            alg_name = algorithm.__class__.__name__
            compute_times = self.cumulative_time[alg_name]
            
            # Calculate marker interval based on number of rounds
            marker_interval = max(1, len(compute_times) // 20)
            
            plt.plot(range(1, len(compute_times) + 1), compute_times, 
                     label=alg_name, marker='o', markersize=4, markevery=marker_interval)
        plt.xlabel('Round')
        plt.ylabel('Compute Time (seconds)')
        plt.title('Algorithm Compute Time over Rounds')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.results_dir}/algorithm_compute_time.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_accuracy_time_ratio(self):
        plt.figure(figsize=(10, 6))
        for algorithm in self.algorithms:
            alg_name = algorithm.__class__.__name__
            accuracies = self.cumulative_accuracy[alg_name]
            compute_times = self.cumulative_time[alg_name]
            ratios = []
            for acc, time in zip(accuracies, compute_times):
                if time > 0:
                    ratios.append(acc / time)
                else:
                    ratios.append(0)  # or you could use float('inf') if you prefer
            
            # Calculate marker interval based on number of rounds
            marker_interval = max(1, len(ratios) // 20)
            
            plt.plot(range(1, len(ratios) + 1), ratios, 
                     label=alg_name, marker='o', markersize=4, markevery=marker_interval)
        plt.xlabel('Round')
        plt.ylabel('Accuracy / Compute Time')
        plt.title('Algorithm Efficiency (Accuracy/Time) over Rounds')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.results_dir}/algorithm_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()

# Example Algorithm Implementation (to be expanded later)
class RandomGuessAlgorithm(Algorithm):
    def predict(self, observation: str) -> int:
        # Randomly guess a label
        return random.randint(0, 4)

class OnlyZerosAlgorithm(Algorithm):
    def predict(self, observation: str) -> int:
        return 0


if __name__ == "__main__":
    problem_setup = ProblemSetupTicTacToe()
    algorithms = [
        RandomGuessAlgorithm(),
        OnlyZerosAlgorithm(),
        # RandomForestAlgorithm(),
        # KNNAlgorithm(),
        # PrototypicalNetworkAlgorithm(),
        # BayesianNNAlgorithm(),
        # LinearRegressionAlgorithm(),
        # SiameseNetworkAlgorithm(),
        # GaussianProcessAlgorithm(),
        # TransformerAlgorithm(),
        # GPT2Algorithm(),
        # ForwardForwardAlgorithm(),
        # DQNAlgorithm(),
        # DiffusionAlgorithm(),
        # DistributionApproximatorAlgorithm(),
        # MultiArmedBanditAlgorithm(),
        # MultiArmedBanditAlgorithm(strategy='thompson_sampling'),
        # MAMLReptileAlgorithm(),
        # # Ours:
        # LocusBanditAlgorithm(),
        #LocusAlgorithm(),
        SATAlgorithm()
    ]
    tester = Tester(problem_setup, algorithms, rounds=30600, ON_THE_HOUSE=True)
    tester.run_tests()
    tester.compute_metrics()
    tester.plot_results()
    print(f"Results and graphs saved in '{tester.results_dir}'")
