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
        self.clauses = clauses  # List of clauses representing the CNF formula
        self.complexity = complexity or sum(len(clause) for clause in clauses)
        self.score = 0
        self.is_active = True
        self.last_evaluated = 0
        # Add Bayesian attributes
        self.prior_prob = None
        self.posterior_prob = None

    def __str__(self):
        return f"Hypothesis(clauses={self.clauses}, complexity={self.complexity}, score={self.score:.2f}, last_evaluated={self.last_evaluated})"

class SATHypothesesAlgorithm(Algorithm):
    def __init__(self, num_inputs=9, input_bits=3, num_outputs=5):
        super().__init__()
        self.num_inputs = num_inputs
        self.input_bits = input_bits
        self.num_outputs = num_outputs
        self.observations = []
        self.round_count = 0
        self.cache = {}
        self.cnf = CNF()
        logger.debug("Initializing SATAlgorithm")
        self.initialize_general_constraints()

        # Hypothesis management
        self.hypotheses = []  # List of active hypotheses
        self.rejected_hypotheses = []
        self.rejected_hypotheses_count = 0  # Initialize the counter
        self.max_clause_size = 2  # Starting with simple hypotheses
        self.hypothesis_generation_limit = 100000  # Limit to avoid combinatorial explosion
        self.min_hypotheses = 10000  # Minimum number of hypotheses to maintain
        
        # Scoring parameters
        self.alpha = 0.5  # Weight for complexity score
        self.beta = 0.5  # Weight for CNF simplification score
        self.hypothesis_database = set()  # For avoiding redundancy
        self.optimal_hypotheses = []  # Will store hypotheses with highest scores
        self.reevaluation_interval = 10  # Reevaluate non-optimal hypotheses every 10 rounds
        self.target_checks_per_round = 1000  # Target number of hypotheses to check per round

        # New attributes for auto-tuning
        self.target_hypotheses_per_round = 200
        self.moving_average_hypotheses = 200  # Initial value
        self.alpha = 0.1  # Smoothing factor for moving average
        self.temperature = 1.0  # Temperature for softmax, will be auto-tuned

        # New attributes for hypothesis generation
        self.new_hypotheses_per_round = 200
        self.current_clause_size = 1
        self.current_combination_index = 0
        self.current_sign_index = 0

        # Add Bayesian parameters
        self.total_hypotheses_count = 0
        self.gamma = 0.4  # Weight for probability score
        self.alpha = 0.3  # Adjusted weight for complexity
        self.beta = 0.3   # Adjusted weight for simplification

        # Update hypothesis generation parameters
        self.max_clause_size = 2  # Starting with simple hypotheses
        self.current_clause_size = 1
        self.max_possible_clause_size = num_inputs * input_bits  # Add this line

    def initialize_general_constraints(self):
        logger.debug("Initializing general constraints")
        if self.cnf is None:
            self.cnf = CNF()
        else:
            self.cnf.clauses.clear()
        
        # Add constraints for each input and output position (exactly one state at a time)
        for i in range(self.num_inputs):
            # At least one state is true
            self.cnf.append([self._var_input_bit(i, b) for b in range(self.input_bits)])
            # At most one state is true
            for b1, b2 in itertools.combinations(range(self.input_bits), 2):
                self.cnf.append([-self._var_input_bit(i, b1), -self._var_input_bit(i, b2)])

        # At least one output state is true
        self.cnf.append([self._var_output_bit(b) for b in range(self.num_outputs)])
        # At most one output state is true
        for b1, b2 in itertools.combinations(range(self.num_outputs), 2):
            self.cnf.append([-self._var_output_bit(b1), -self._var_output_bit(b2)])
        
        logger.debug(f"General constraints initialized. Total clauses: {len(self.cnf.clauses)}")

    def _var_input_bit(self, i, b):
        return i * self.input_bits + b + 1

    def _var_output_bit(self, b):
        return self.num_inputs * self.input_bits + b + 1

    def _convert_to_binary(self, data_str, mapping):
        """Convert input/output strings to binary representation using provided mapping."""
        binary = ''
        for char in str(data_str):  # Convert to string to handle both str and int inputs
            binary += mapping.get(char, '000')
        return binary

    def _input_to_binary(self, input_str):
        """Convert input string to binary representation."""
        mapping = {
            '0': '100',
            '1': '010',
            '2': '001'
        }
        return self._convert_to_binary(input_str, mapping)

    def _output_to_binary(self, output):
        """Convert output value to binary representation."""
        mapping = {str(i): '1' if i == output else '0' for i in range(self.num_outputs)}
        return self._convert_to_binary(output, mapping)

    def _binary_to_output(self, binary):
        """Convert binary string back to output value."""
        return binary.index('1')

    def remove_subsumed_clauses(self, cnf):
        original_clause_count = len(cnf.clauses)
        
        # Convert to frozenset for faster operations and deduplication
        clause_sets = [frozenset(clause) for clause in cnf.clauses]
        unique_clauses = set(clause_sets)
        
        # Create list of non-subsumed clauses
        non_subsumed = []
        for clause1 in unique_clauses:
            is_subsumed = False
            for clause2 in unique_clauses:
                if clause1 != clause2 and clause1.issubset(clause2):
                    is_subsumed = True
                    break
            if not is_subsumed:
                non_subsumed.append(list(clause1))
        
        # Create new CNF with simplified clauses
        simplified_cnf = CNF(from_clauses=non_subsumed)
        removed_clauses = original_clause_count - len(simplified_cnf.clauses)
        
        # Maintain existing printouts
        #print(f"Clause simplification: Removed {removed_clauses} redundant clauses")
        #print(f"Original clauses: {original_clause_count}, Simplified clauses: {len(simplified_cnf.clauses)}")
        
        return simplified_cnf
    

    def generate_hypotheses(self):
        new_hypotheses = []
        input_variables = list(range(1, self.num_inputs * self.input_bits + 1))
        output_variables = list(range(self.num_inputs * self.input_bits + 1, self.num_inputs * self.input_bits + self.num_outputs + 1))

        while len(new_hypotheses) < self.new_hypotheses_per_round:
            if self.current_clause_size > self.max_clause_size:
                break

            # Generate multiple clauses for each hypothesis
            num_clauses = random.randint(1, 3)  # Generate 1 to 3 clauses per hypothesis
            hypothesis_clauses = []

            for _ in range(num_clauses):
                input_clause = self.generate_input_clause(input_variables)
                output_literal = self.generate_output_literal(output_variables)
                implication = [-lit for lit in input_clause] + [output_literal]
                hypothesis_clauses.append(implication)

            hypothesis_key = tuple(sorted(tuple(sorted(clause)) for clause in hypothesis_clauses))
            if hypothesis_key not in self.hypothesis_database:
                self.hypothesis_database.add(hypothesis_key)
                new_hypothesis = Hypothesis(clauses=hypothesis_clauses)
                new_hypotheses.append(new_hypothesis)

                if len(new_hypotheses) >= self.new_hypotheses_per_round:
                    return new_hypotheses

            self.update_clause_generation_state()

        for hypothesis in new_hypotheses:
            # Assign initial prior probability
            hypothesis.prior_prob = 1.0 / (self.total_hypotheses_count + len(new_hypotheses))
            hypothesis.posterior_prob = hypothesis.prior_prob

        self.total_hypotheses_count += len(new_hypotheses)
        return new_hypotheses

    def generate_input_clause(self, input_variables):
        # Ensure clause size doesn't exceed available variables
        effective_size = min(self.current_clause_size, len(input_variables))
        combination = random.sample(input_variables, effective_size)
        signs = [random.choice([True, False]) for _ in range(effective_size)]
        return [var if sign else -var for var, sign in zip(combination, signs)]

    def generate_output_literal(self, output_variables):
        output_var = random.choice(output_variables)
        output_sign = random.choice([True, False])
        return output_var if output_sign else -output_var

    def update_clause_generation_state(self):
        max_possible_size = self.num_inputs * self.input_bits  # Maximum possible clause size
        
        self.current_clause_size += 1
        if self.current_clause_size > self.max_clause_size:
            self.max_clause_size = min(self.max_clause_size + 1, max_possible_size)
            self.current_clause_size = 1

    def compute_scores(self, hypothesis, simplified_cnf):
        """Precompute all scoring components for a hypothesis."""
        hypothesis.complexity_score = 1 / hypothesis.complexity
        hypothesis.simplification_score = 1 / len(simplified_cnf.clauses)
        hypothesis.probability_score = hypothesis.posterior_prob or 0.0
        
        # Calculate final score using weights
        hypothesis.score = (
            self.alpha * hypothesis.complexity_score + 
            self.beta * hypothesis.simplification_score + 
            self.gamma * hypothesis.probability_score
        )
        return hypothesis.score

    def validate_hypotheses(self):
        """Main validation flow coordinating hypothesis evaluation and updates."""
        all_hypotheses = self.hypotheses + self.generate_hypotheses()
        base_cnf = self.get_base_cnf()
        selected_hypotheses = self.select_hypotheses_to_evaluate(all_hypotheses)
        valid_hypotheses, rejected_hypotheses = self.evaluate_hypotheses(selected_hypotheses, base_cnf)
        
        self.update_hypothesis_lists(valid_hypotheses, rejected_hypotheses)
        self.update_optimal_hypotheses(valid_hypotheses)
        self.update_moving_average(len(selected_hypotheses))
        self.auto_tune_temperature()
        
        self.print_validation_results(len(valid_hypotheses), len(rejected_hypotheses), len(selected_hypotheses))

    def get_base_cnf(self):
        """Creates base CNF with current constraints."""
        base_cnf = CNF()
        base_cnf.extend(self.cnf.clauses)
        return base_cnf

    def select_hypotheses_to_evaluate(self, all_hypotheses):
        """Selects subset of hypotheses for evaluation using temperature-based sampling."""
        if not all_hypotheses:
            print("No hypotheses to validate.")
            return []

        # Calculate selection probabilities using softmax
        scores = np.array([max(h.score, 0.001) for h in all_hypotheses])
        probabilities = np.exp(scores / self.temperature) / np.sum(np.exp(scores / self.temperature))
        
        # Calculate number of hypotheses to evaluate
        target_ratio = self.target_hypotheses_per_round / self.moving_average_hypotheses
        num_to_evaluate = min(
            len(all_hypotheses), 
            max(10, int(self.moving_average_hypotheses * target_ratio))
        )
        
        # Select hypotheses using weighted sampling
        selected_indices = np.random.choice(
            len(all_hypotheses),
            size=num_to_evaluate,
            replace=False,
            p=probabilities
        )
        return [all_hypotheses[i] for i in selected_indices]

    def evaluate_hypotheses(self, hypotheses, base_cnf):
        """Evaluates each hypothesis and returns valid and rejected lists."""
        valid_hypotheses = []
        rejected_hypotheses = []

        for hypothesis in hypotheses:
            temp_cnf = CNF()
            temp_cnf.extend(base_cnf.clauses)
            temp_cnf.extend(hypothesis.clauses)

            with Solver(bootstrap_with=temp_cnf.clauses) as solver:
                if solver.solve():
                    simplified_cnf = self.remove_subsumed_clauses(temp_cnf)
                    self.compute_scores(hypothesis, simplified_cnf)
                    hypothesis.last_evaluated = self.round_count
                    valid_hypotheses.append(hypothesis)
                else:
                    hypothesis.is_active = False
                    rejected_hypotheses.append(hypothesis)

        return valid_hypotheses, rejected_hypotheses

    def update_hypothesis_lists(self, valid_hypotheses, rejected_hypotheses):
        """Updates the main hypothesis lists with evaluation results."""
        self.hypotheses = [h for h in self.hypotheses if h.is_active] + valid_hypotheses
        self.rejected_hypotheses.extend(rejected_hypotheses)
        self.rejected_hypotheses_count += len(rejected_hypotheses)

    def update_optimal_hypotheses(self, valid_hypotheses):
        """Updates the list of optimal hypotheses."""
        self.optimal_hypotheses = sorted(
            valid_hypotheses, 
            key=lambda h: h.score, 
            reverse=True
        )[:10]

    def update_moving_average(self, num_evaluated):
        """Updates the moving average of hypotheses evaluated per round."""
        self.moving_average_hypotheses = (
            (1 - self.alpha) * self.moving_average_hypotheses + 
            self.alpha * num_evaluated
        )

    def auto_tune_temperature(self):
        """Adjusts the temperature parameter based on moving average."""
        if self.moving_average_hypotheses > self.target_hypotheses_per_round:
            self.temperature *= 1.1
        else:
            self.temperature *= 0.9
        self.temperature = max(0.01, min(100, self.temperature))

    def print_validation_results(self, num_valid, num_rejected, num_evaluated):
        """Prints the results of hypothesis validation."""
        print(f"Validated hypotheses. Active: {len(self.hypotheses)}, Rejected this round: {num_rejected}")
        print(f"Total rejected hypotheses: {self.rejected_hypotheses_count}")
        print(f"Hypotheses checked this round: {num_evaluated}")
        print(f"Moving average hypotheses per round: {self.moving_average_hypotheses:.2f}")
        print(f"Current temperature: {self.temperature:.2f}")
        print(f"Top 5 hypotheses scores: {[h.score for h in self.optimal_hypotheses[:5]]}")

    def update_hypotheses(self, observation, correct_label):
        self.validate_hypotheses()
        self.print_hypothesis_stats()

    def _add_variable_constraints(self, binary_str, var_func):
        """Helper method to generate variable constraints from binary string."""
        clauses = []
        for i, bit in enumerate(binary_str):
            var = var_func(i)
            clauses.append([var] if bit == '1' else [-var])
        return clauses

    def _get_input_constraints(self, input_binary):
        """Generate input constraints from binary input string."""
        return self._add_variable_constraints(
            input_binary,
            lambda i: self._var_input_bit(i // self.input_bits, i % self.input_bits)
        )

    def _get_output_constraints(self, output_binary):
        """Generate output constraints from binary output string."""
        return self._add_variable_constraints(
            output_binary,
            self._var_output_bit
        )

    def predict_basic(self, observation: str) -> int:
        logger.debug(f"Predicting for observation: {observation}")
        if self.cnf is None:
            logger.error("self.cnf is None in predict_basic")
            self.cnf = CNF()
            self.initialize_general_constraints()

        if observation in self.cache:
            print(f"Cache hit for observation: {observation}")
            return self.cache[observation]

        input_binary = self._input_to_binary(observation)
        input_clauses = self._get_input_constraints(input_binary)

        # Include the observation history and problem constraints
        base_cnf = CNF()
        base_cnf.extend(self.cnf.clauses)
        base_cnf.extend(input_clauses)

        # Group hypotheses by score
        score_groups = defaultdict(list)
        for hypothesis in self.hypotheses:
            if hypothesis.is_active:
                score_groups[hypothesis.score].append(hypothesis)

        # Print out the set of possible hypotheses
        print("Set of possible hypotheses for prediction:")
        count = 0
        for score, hypotheses in sorted(score_groups.items(), reverse=True):
            print(f"Score {score:.4f}: {len(hypotheses)} hypotheses")
            for h in hypotheses[:5]:  # Print details of up to 5 hypotheses per score group
                print(f"  {h}")
            if len(hypotheses) > 5:
                print(f"  ... and {len(hypotheses) - 5} more")
            if count > 10:
                break
            count += 1

        # Select the group with the highest score
        if score_groups:
            max_score = max(score_groups.keys())
            best_hypotheses = score_groups[max_score]

            print(f"\nSelecting from {len(best_hypotheses)} hypotheses with highest score {max_score:.4f}")

            # Randomly select a hypothesis from the best group
            selected_hypothesis = random.choice(best_hypotheses)

            print(f"Selected hypothesis: {selected_hypothesis}")

            # Create a CNF for the selected hypothesis
            prediction_cnf = CNF()
            prediction_cnf.extend(base_cnf.clauses)
            prediction_cnf.extend(selected_hypothesis.clauses)

            with Solver(bootstrap_with=prediction_cnf.clauses) as solver:
                is_satisfiable = solver.solve()
                if is_satisfiable:
                    model = solver.get_model()
                    output_bits = ''.join(['1' if self._var_output_bit(b) in model else '0' for b in range(self.num_outputs)])
                    result = self._binary_to_output(output_bits)
                    print(f"Hypothesis with score {selected_hypothesis.score:.4f} predicts output: {result}")
                    # Store the result in the cache before returning
                    self.cache[observation] = result
                    return result

        # If no hypothesis provides a solution, handle accordingly
        print("Error: No hypotheses provide a valid prediction.")
        # Optionally, generate new hypotheses or return a default prediction
        default_output = random.randint(0, self.num_outputs - 1)
        self.cache[observation] = default_output
        return default_output

    def update_history(self, observation: str, guess: int, correct_label: int):
        logger.debug(f"Updating history: observation={observation}, guess={guess}, correct_label={correct_label}")
        if self.cnf is None:
            logger.error("self.cnf is None in update_history")
            self.cnf = CNF()
            self.initialize_general_constraints()

        super().update_history(observation, guess, correct_label)
        self.cache[observation] = correct_label

        input_binary = self._input_to_binary(observation)
        self.observations.append((input_binary, correct_label))

        # Get input and output constraints
        input_vars = self._get_input_constraints(input_binary)
        output_binary = self._output_to_binary(correct_label)
        output_vars = self._get_output_constraints(output_binary)

        # Flatten the constraints for the CNF
        input_literals = [clause[0] for clause in input_vars]  # Each clause has one literal
        output_literals = [clause[0] for clause in output_vars]  # Each clause has one literal

        # Add the implication constraint
        self.cnf.append([-v for v in input_literals] + output_literals)

        print(f"New observation: input={input_binary}, output={correct_label}")
        print(f"Total observations: {len(self.observations)}")
        print(f"Cache size: {len(self.cache)}")
        print(f"Total clauses: {len(self.cnf.clauses)}")

        self.round_count += 1
        self.update_hypotheses(observation, correct_label)

        if self.round_count % 100 == 0:
            print("Performing clause reduction...")
            self.cnf = self.remove_subsumed_clauses(self.cnf)
            print("Clause reduction complete.")
            print(f"Total clauses: {len(self.cnf.clauses)}")
        else:
            print("Skipping clause reduction this round.")

    def print_hypothesis_stats(self):
        if not self.hypotheses:
            print("No active hypotheses.")
            return

        ages = [self.round_count - h.last_evaluated for h in self.hypotheses]
        scores = [h.score for h in self.hypotheses]

        print("\nHypothesis Statistics:")
        print(f"Total active hypotheses: {len(self.hypotheses)}")
        print(f"Total rejected hypotheses: {self.rejected_hypotheses_count}")
        print(f"Age range: {min(ages)} - {max(ages)} rounds")
        print(f"Age mean: {sum(ages) / len(ages):.2f} rounds")
        print(f"Age median: {sorted(ages)[len(ages)//2]} rounds")
        print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"Score mean: {sum(scores) / len(scores):.4f}")
        print(f"Score median: {sorted(scores)[len(scores)//2]:.4f}")
        
        # Calculate age distribution
        age_distribution = {}
        for age in ages:
            age_distribution[age] = age_distribution.get(age, 0) + 1
        
        print("\nAge Distribution (top 10):")
        for age, count in sorted(age_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"Age {age}: {count} hypotheses")
        
        # Calculate score distribution
        score_distribution = {}
        for score in scores:
            score_distribution[score] = score_distribution.get(score, 0) + 1
        
        print("\nScore Distribution (top 10):")
        for score, count in sorted(score_distribution.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"Score {score:.4f}: {count} hypotheses")

        print("\nTop 5 Hypotheses:")
        for i, h in enumerate(sorted(self.hypotheses, key=lambda x: x.score, reverse=True)[:5], 1):
            print(f"{i}. Score: {h.score:.4f}, Age: {self.round_count - h.last_evaluated}, Complexity: {h.complexity}, Clauses: {h.clauses}")

        print("\n" + "="*50)

    def predict_with_cache(self, observation: str) -> int:
        return self.predict_basic(observation)

    def predict(self, observation: str) -> int:
        return self.predict_basic(observation)

    def update_hypotheses_probabilities(self, observation, correct_label):
        evidence_prob = 0.0
        
        # Update posterior probabilities using Bayes' theorem
        for hypothesis in self.hypotheses:
            if not hypothesis.is_active:
                continue
            
            likelihood = self.compute_likelihood(hypothesis, observation, correct_label)
            hypothesis.posterior_prob = hypothesis.prior_prob * likelihood
            evidence_prob += hypothesis.posterior_prob
        
        # Normalize and update priors
        if evidence_prob > 0:
            for hypothesis in self.hypotheses:
                if hypothesis.is_active:
                    hypothesis.posterior_prob /= evidence_prob
                    hypothesis.prior_prob = hypothesis.posterior_prob

    def compute_likelihood(self, hypothesis, observation, correct_label):
        temp_cnf = CNF()
        temp_cnf.extend(self.cnf.clauses)
        temp_cnf.extend(hypothesis.clauses)
        
        # Add observation constraints
        input_binary = self._input_to_binary(observation)
        output_binary = self._output_to_binary(correct_label)
        
        for i, bit in enumerate(input_binary):
            var = self._var_input_bit(i // self.input_bits, i % self.input_bits)
            temp_cnf.append([var] if bit == '1' else [-var])
        
        for b, bit in enumerate(output_binary):
            var = self._var_output_bit(b)
            temp_cnf.append([var] if bit == '1' else [-var])
        
        with Solver(bootstrap_with=temp_cnf.clauses) as solver:
            return 1.0 if solver.solve() else 0.0

    def compute_bayesian_surprise(self):
        active_hypotheses = [h for h in self.hypotheses if h.is_active]
        if not active_hypotheses:
            return 0.0
            
        prior_probs = np.array([h.prior_prob for h in active_hypotheses])
        posterior_probs = np.array([h.posterior_prob for h in active_hypotheses])
        
        # Compute KL divergence with numerical stability
        kl_divergence = np.sum(
            posterior_probs * (
                np.log(posterior_probs + 1e-12) - 
                np.log(prior_probs + 1e-12)
            )
        )
        return kl_divergence

