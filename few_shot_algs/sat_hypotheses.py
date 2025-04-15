import itertools
from pysat.formula import CNF
from pysat.solvers import Solver
from few_shot_algs.few_shot_alg import Algorithm
import random
import time
import sys
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Hypothesis:
    def __init__(self, clauses, complexity=None):
        self.clauses = clauses
        self.complexity = complexity or sum(len(clause) for clause in clauses)
        self.score = 0
        self.is_active = True
        self.last_evaluated = 0
        self.prior_prob = None
        self.posterior_prob = None
        # Scoring components
        self.complexity_score = 0
        self.simplification_score = 0
        self.probability_score = 0

    def __str__(self):
        return f"Hypothesis(clauses={self.clauses}, complexity={self.complexity}, score={self.score:.2f})"


class CNFManager:
    def __init__(self, num_inputs, input_bits, num_outputs):
        self.num_inputs = num_inputs
        self.input_bits = input_bits
        self.num_outputs = num_outputs
        self.cnf = CNF()
        self.initialize_constraints()

    def initialize_constraints(self):
        # Add constraints for inputs
        for i in range(self.num_inputs):
            self.cnf.append([self._var_input_bit(i, b) for b in range(self.input_bits)])
            for b1, b2 in itertools.combinations(range(self.input_bits), 2):
                self.cnf.append([-self._var_input_bit(i, b1), -self._var_input_bit(i, b2)])

        # Add constraints for outputs
        self.cnf.append([self._var_output_bit(b) for b in range(self.num_outputs)])
        for b1, b2 in itertools.combinations(range(self.num_outputs), 2):
            self.cnf.append([-self._var_output_bit(b1), -self._var_output_bit(b2)])

    def _var_input_bit(self, i, b):
        return i * self.input_bits + b + 1

    def _var_output_bit(self, b):
        return self.num_inputs * self.input_bits + b + 1

    def get_input_constraints(self, input_binary):
        return [
            [self._var_input_bit(i // self.input_bits, i % self.input_bits)] 
            if bit == '1' else 
            [-self._var_input_bit(i // self.input_bits, i % self.input_bits)]
            for i, bit in enumerate(input_binary)
        ]

    def get_output_constraints(self, output_binary):
        return [
            [self._var_output_bit(b)] if bit == '1' else [-self._var_output_bit(b)]
            for b, bit in enumerate(output_binary)
        ]

    def remove_subsumed_clauses(self):
        clause_sets = [frozenset(clause) for clause in self.cnf.clauses]
        unique_clauses = set(clause_sets)
        
        non_subsumed = [
            list(clause1) for clause1 in unique_clauses
            if not any(
                clause1 != clause2 and clause1.issubset(clause2)
                for clause2 in unique_clauses
            )
        ]
        
        self.cnf = CNF(from_clauses=non_subsumed)
        return self.cnf

class HypothesisGenerator:
    def __init__(self, num_inputs, input_bits, num_outputs, new_hypotheses_per_round=200):
        self.num_inputs = num_inputs
        self.input_bits = input_bits
        self.num_outputs = num_outputs
        self.new_hypotheses_per_round = new_hypotheses_per_round
        self.hypothesis_database = set()
        self.current_clause_size = 1
        self.max_clause_size = 2
        self.max_possible_clause_size = num_inputs * input_bits

    def generate_hypotheses(self):
        new_hypotheses = []
        input_variables = list(range(1, self.num_inputs * self.input_bits + 1))
        output_variables = list(range(self.num_inputs * self.input_bits + 1, 
                                    self.num_inputs * self.input_bits + self.num_outputs + 1))

        while len(new_hypotheses) < self.new_hypotheses_per_round:
            if self.current_clause_size > self.max_clause_size:
                break

            hypothesis_clauses = [
                self._generate_clause(input_variables, output_variables)
                for _ in range(random.randint(1, 3))
            ]

            hypothesis_key = tuple(sorted(tuple(sorted(clause)) for clause in hypothesis_clauses))
            if hypothesis_key not in self.hypothesis_database:
                self.hypothesis_database.add(hypothesis_key)
                new_hypotheses.append(Hypothesis(clauses=hypothesis_clauses))

            self._update_clause_generation_state()

        return new_hypotheses

    def _generate_clause(self, input_variables, output_variables):
        input_clause = self._generate_input_clause(input_variables)
        output_literal = self._generate_output_literal(output_variables)
        return [-lit for lit in input_clause] + [output_literal]

    def _generate_input_clause(self, input_variables):
        effective_size = min(self.current_clause_size, len(input_variables))
        combination = random.sample(input_variables, effective_size)
        return [var if random.choice([True, False]) else -var for var in combination]

    def _generate_output_literal(self, output_variables):
        output_var = random.choice(output_variables)
        return output_var if random.choice([True, False]) else -output_var

    def _update_clause_generation_state(self):
        self.current_clause_size += 1
        if self.current_clause_size > self.max_clause_size:
            self.max_clause_size = min(self.max_clause_size + 1, self.max_possible_clause_size)
            self.current_clause_size = 1

class HypothesisValidator:
    def __init__(self, temperature=1.0, target_hypotheses_per_round=200):
        self.temperature = temperature
        self.target_hypotheses_per_round = target_hypotheses_per_round
        self.moving_average_hypotheses = target_hypotheses_per_round
        self.alpha = 0.1  # Smoothing factor

    def validate_hypotheses(self, all_hypotheses, base_cnf):
        selected_hypotheses = self._select_hypotheses(all_hypotheses)
        return self._evaluate_hypotheses(selected_hypotheses, base_cnf)

    def _select_hypotheses(self, all_hypotheses):
        if not all_hypotheses:
            return []

        scores = np.array([max(h.score, 0.001) for h in all_hypotheses])
        probabilities = np.exp(scores / self.temperature) / np.sum(np.exp(scores / self.temperature))
        
        target_ratio = self.target_hypotheses_per_round / self.moving_average_hypotheses
        num_to_evaluate = min(len(all_hypotheses), 
                            max(10, int(self.moving_average_hypotheses * target_ratio)))
        
        selected_indices = np.random.choice(len(all_hypotheses), 
                                          size=num_to_evaluate,
                                          replace=False,
                                          p=probabilities)
        return [all_hypotheses[i] for i in selected_indices]

    def _evaluate_hypotheses(self, hypotheses, base_cnf):
        valid_hypotheses = []
        rejected_hypotheses = []

        for hypothesis in hypotheses:
            temp_cnf = CNF()
            temp_cnf.extend(base_cnf.clauses)
            temp_cnf.extend(hypothesis.clauses)

            with Solver(bootstrap_with=temp_cnf.clauses) as solver:
                if solver.solve():
                    valid_hypotheses.append(hypothesis)
                else:
                    hypothesis.is_active = False
                    rejected_hypotheses.append(hypothesis)

        return valid_hypotheses, rejected_hypotheses

    def update_moving_average(self, num_evaluated):
        self.moving_average_hypotheses = (1 - self.alpha) * self.moving_average_hypotheses + self.alpha * num_evaluated

    def auto_tune_temperature(self):
        if self.moving_average_hypotheses > self.target_hypotheses_per_round:
            self.temperature *= 1.1
        else:
            self.temperature *= 0.9
        self.temperature = max(0.01, min(100, self.temperature))

class HypothesisScorer:
    def __init__(self, alpha=0.3, beta=0.3, gamma=0.4):
        self.alpha = alpha  # Weight for complexity
        self.beta = beta    # Weight for simplification
        self.gamma = gamma  # Weight for probability

    def compute_scores(self, hypothesis, simplified_cnf):
        hypothesis.complexity_score = 1 / hypothesis.complexity
        hypothesis.simplification_score = 1 / len(simplified_cnf.clauses)
        hypothesis.probability_score = hypothesis.posterior_prob or 0.0
        
        hypothesis.score = (
            self.alpha * hypothesis.complexity_score + 
            self.beta * hypothesis.simplification_score + 
            self.gamma * hypothesis.probability_score
        )
        return hypothesis.score

class BinaryConverter:
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs
        self.input_mapping = {'0': '100', '1': '010', '2': '001'}

    def input_to_binary(self, input_str):
        return ''.join(self.input_mapping.get(char, '000') for char in str(input_str))

    def output_to_binary(self, output):
        return ''.join('1' if i == output else '0' for i in range(self.num_outputs))

    def binary_to_output(self, binary):
        return binary.index('1')

class HypothesisStats:
    def __init__(self):
        self.round_count = 0
        self.hypotheses = []
        self.rejected_hypotheses_count = 0

    def print_stats(self, hypotheses):
        if not hypotheses:
            print("No active hypotheses.")
            return

        ages = [self.round_count - h.last_evaluated for h in hypotheses]
        scores = [h.score for h in hypotheses]

        self._print_basic_stats(hypotheses, ages, scores)
        self._print_age_distribution(ages)
        self._print_score_distribution(scores)
        self._print_top_hypotheses(hypotheses)
        print("\n" + "="*50)

    def _print_basic_stats(self, hypotheses, ages, scores):
        print("\nHypothesis Statistics:")
        print(f"Total active hypotheses: {len(hypotheses)}")
        print(f"Total rejected hypotheses: {self.rejected_hypotheses_count}")
        print(f"Age range: {min(ages)} - {max(ages)} rounds")
        print(f"Age mean: {sum(ages) / len(ages):.2f} rounds")
        print(f"Age median: {sorted(ages)[len(ages)//2]} rounds")
        print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"Score mean: {sum(scores) / len(scores):.4f}")
        print(f"Score median: {sorted(scores)[len(scores)//2]:.4f}")

    def _print_age_distribution(self, ages):
        age_distribution = {}
        for age in ages:
            age_distribution[age] = age_distribution.get(age, 0) + 1
        
        print("\nAge Distribution (top 10):")
        for age, count in sorted(age_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"Age {age}: {count} hypotheses")

    def _print_score_distribution(self, scores):
        score_distribution = {}
        for score in scores:
            score_distribution[score] = score_distribution.get(score, 0) + 1
        
        print("\nScore Distribution (top 10):")
        for score, count in sorted(score_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"Score {score:.4f}: {count} hypotheses")

    def _print_top_hypotheses(self, hypotheses):
        print("\nTop 5 Hypotheses:")
        for i, h in enumerate(sorted(hypotheses, key=lambda x: x.score, reverse=True)[:5], 1):
            print(f"{i}. Score: {h.score:.4f}, Age: {self.round_count - h.last_evaluated}, "
                  f"Complexity: {h.complexity}, Clauses: {h.clauses}")

class PredictionManager:
    def __init__(self, num_outputs, cnf_manager):
        self.num_outputs = num_outputs
        self.cnf_manager = cnf_manager
        self.cache = {}

    def predict(self, observation, input_binary, hypotheses):
        if observation in self.cache:
            print(f"Cache hit for observation: {observation}")
            return self.cache[observation]

        input_clauses = self.cnf_manager.get_input_constraints(input_binary)
        score_groups = self._group_hypotheses_by_score(hypotheses)
        
        self._print_hypothesis_groups(score_groups)

        if not score_groups:
            return self._handle_no_hypotheses(observation)

        selected_hypothesis = self._select_best_hypothesis(score_groups)
        result = self._make_prediction(selected_hypothesis, input_clauses, observation)
        return result

    def _group_hypotheses_by_score(self, hypotheses):
        score_groups = defaultdict(list)
        for hypothesis in hypotheses:
            if hypothesis.is_active:
                score_groups[hypothesis.score].append(hypothesis)
        return score_groups

    def _print_hypothesis_groups(self, score_groups):
        print("Set of possible hypotheses for prediction:")
        count = 0
        for score, hypotheses in sorted(score_groups.items(), reverse=True):
            print(f"Score {score:.4f}: {len(hypotheses)} hypotheses")
            for h in hypotheses[:5]:
                print(f"  {h}")
            if len(hypotheses) > 5:
                print(f"  ... and {len(hypotheses) - 5} more")
            if count > 10:
                break
            count += 1

    def _handle_no_hypotheses(self, observation):
        default_output = random.randint(0, self.num_outputs - 1)
        self.cache[observation] = default_output
        return default_output

    def _select_best_hypothesis(self, score_groups):
        max_score = max(score_groups.keys())
        best_hypotheses = score_groups[max_score]
        print(f"\nSelecting from {len(best_hypotheses)} hypotheses with highest score {max_score:.4f}")
        selected_hypothesis = random.choice(best_hypotheses)
        print(f"Selected hypothesis: {selected_hypothesis}")
        return selected_hypothesis

    def _make_prediction(self, hypothesis, input_clauses, observation):
        prediction_cnf = CNF()
        prediction_cnf.extend(self.cnf_manager.cnf.clauses)
        prediction_cnf.extend(input_clauses)
        prediction_cnf.extend(hypothesis.clauses)

        with Solver(bootstrap_with=prediction_cnf.clauses) as solver:
            if solver.solve():
                model = solver.get_model()
                output_bits = ''.join(
                    ['1' if self.cnf_manager._var_output_bit(b) in model else '0' 
                     for b in range(self.num_outputs)]
                )
                result = output_bits.index('1')
                print(f"Hypothesis with score {hypothesis.score:.4f} predicts output: {result}")
                self.cache[observation] = result
                return result

        return self._handle_no_hypotheses(observation)

class SATHypothesesAlgorithm(Algorithm):
    def __init__(self, num_inputs=9, input_bits=3, num_outputs=5):
        super().__init__()
        self.num_inputs = num_inputs
        self.input_bits = input_bits
        self.num_outputs = num_outputs
        self.observations = []
        
        # Initialize managers and helpers
        self.cnf_manager = CNFManager(num_inputs, input_bits, num_outputs)
        self.generator = HypothesisGenerator(num_inputs, input_bits, num_outputs)
        self.validator = HypothesisValidator()
        self.scorer = HypothesisScorer()
        self.binary_converter = BinaryConverter(num_outputs)
        self.stats = HypothesisStats()
        self.prediction_manager = PredictionManager(num_outputs, self.cnf_manager)
        
        # State management
        self.hypotheses = []
        self.rejected_hypotheses = []
        self.optimal_hypotheses = []
        
        logger.debug("Initializing SATAlgorithm")

    def validate_hypotheses(self):
        all_hypotheses = self.hypotheses + self.generator.generate_hypotheses()
        valid_hypotheses, rejected_hypotheses = self.validator.validate_hypotheses(
            all_hypotheses, 
            self.cnf_manager.cnf
        )
        
        for hypothesis in valid_hypotheses:
            simplified_cnf = self.cnf_manager.remove_subsumed_clauses()
            self.scorer.compute_scores(hypothesis, simplified_cnf)
            hypothesis.last_evaluated = self.stats.round_count
        
        self.update_hypothesis_lists(valid_hypotheses, rejected_hypotheses)
        self.validator.update_moving_average(len(valid_hypotheses))
        self.validator.auto_tune_temperature()
        self.print_validation_results(len(valid_hypotheses), len(rejected_hypotheses))

    def update_hypothesis_lists(self, valid_hypotheses, rejected_hypotheses):
        self.hypotheses = [h for h in self.hypotheses if h.is_active] + valid_hypotheses
        self.rejected_hypotheses.extend(rejected_hypotheses)
        self.stats.rejected_hypotheses_count += len(rejected_hypotheses)
        self.optimal_hypotheses = sorted(valid_hypotheses, key=lambda h: h.score, reverse=True)[:10]

    def print_validation_results(self, num_valid, num_rejected):
        print(f"Validated hypotheses. Active: {len(self.hypotheses)}, Rejected this round: {num_rejected}")
        print(f"Total rejected hypotheses: {self.stats.rejected_hypotheses_count}")
        print(f"Hypotheses checked this round: {num_valid}")
        print(f"Moving average hypotheses per round: {self.validator.moving_average_hypotheses:.2f}")
        print(f"Current temperature: {self.validator.temperature:.2f}")
        print(f"Top 5 hypotheses scores: {[h.score for h in self.optimal_hypotheses[:5]]}")

    def predict(self, observation: str) -> int:
        logger.debug(f"Predicting for observation: {observation}")
        input_binary = self.binary_converter.input_to_binary(observation)
        return self.prediction_manager.predict(observation, input_binary, self.hypotheses)

    def update_history(self, observation: str, guess: int, correct_label: int):
        logger.debug(f"Updating history: observation={observation}, guess={guess}, correct_label={correct_label}")
        super().update_history(observation, guess, correct_label)
        self.prediction_manager.cache[observation] = correct_label

        input_binary = self.binary_converter.input_to_binary(observation)
        self.observations.append((input_binary, correct_label))

        input_vars = self.cnf_manager.get_input_constraints(input_binary)
        output_binary = self.binary_converter.output_to_binary(correct_label)
        output_vars = self.cnf_manager.get_output_constraints(output_binary)

        input_literals = [clause[0] for clause in input_vars]
        output_literals = [clause[0] for clause in output_vars]
        self.cnf_manager.cnf.append([-v for v in input_literals] + output_literals)

        print(f"New observation: input={input_binary}, output={correct_label}")
        print(f"Total observations: {len(self.observations)}")
        print(f"Cache size: {len(self.prediction_manager.cache)}")
        print(f"Total clauses: {len(self.cnf_manager.cnf.clauses)}")

        self.stats.round_count += 1
        self.validate_hypotheses()
        self.stats.print_stats(self.hypotheses)

        if self.stats.round_count % 100 == 0:
            print("Performing clause reduction...")
            self.cnf_manager.remove_subsumed_clauses()
            print("Clause reduction complete.")
            print(f"Total clauses: {len(self.cnf_manager.cnf.clauses)}")
        else:
            print("Skipping clause reduction this round.")

