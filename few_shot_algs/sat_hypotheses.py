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
from typing import List, Tuple

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
        self.current_clause_size = 1 # Start with simplest clauses
        self.max_clause_size = 4     # Start allowing more complex clauses sooner
        self.max_possible_clause_size = num_inputs * input_bits # Absolute max
        # Store history reference for guided generation
        self.history = [] # This needs to be updated by the main algorithm

    def set_history(self, history):
        """Allows the main algorithm to provide the history."""
        self.history = history

    def generate_hypotheses(self):
        """Generates hypotheses, guided by history if available."""
        if not self.history:
            logger.warning("HypothesisGenerator generating random hypotheses (no history)." )
            return self._generate_random_hypotheses()
        else:
            return self._generate_guided_hypotheses()

    def _generate_guided_hypotheses(self):
        """Generates hypotheses by combining clauses from multiple observations."""
        new_hypotheses = []
        attempts = 0
        max_attempts = self.new_hypotheses_per_round * 20 # Increased attempts

        while len(new_hypotheses) < self.new_hypotheses_per_round and attempts < max_attempts:
            attempts += 1
            # Combine clauses from 5 to 10 distinct observations (previously 2-3)
            num_clauses_per_hyp = random.randint(5, 10)
            if len(self.history) < num_clauses_per_hyp:
                if len(self.history) == 0:
                    logger.warning("Guided generation called with empty history!")
                    break # Cannot generate anything
                num_clauses_per_hyp = len(self.history)
                if num_clauses_per_hyp == 0: continue

            # Sample distinct observations from history
            try:
                clause_generating_indices = random.sample(range(len(self.history)), num_clauses_per_hyp)
            except ValueError:
                # Handle case where history size < num_clauses_per_hyp after check (shouldn't happen often)
                logger.warning(f"Could not sample {num_clauses_per_hyp} indices from history of size {len(self.history)}.")
                continue

            hypothesis_clauses = []
            total_complexity = 0
            possible_to_generate = True
            for obs_idx in clause_generating_indices:
                input_assumptions, correct_output_lit = self.history[obs_idx]

                # Generate a clause that supports this observation
                # Use current_clause_size to determine complexity of each sub-clause
                num_input_lits = min(self.current_clause_size, len(input_assumptions))
                if num_input_lits <= 0:
                    logger.warning(f"Skipping obs {obs_idx} in guided generation: No input assumptions.")
                    possible_to_generate = False
                    break
                relevant_input_lits = random.sample(input_assumptions, num_input_lits)

                # CNF form: [-relevant_input_lit1, ..., correct_output_lit]
                clause = sorted([-lit for lit in relevant_input_lits] + [correct_output_lit])
                hypothesis_clauses.append(clause)
                total_complexity += len(clause)

            if not possible_to_generate or not hypothesis_clauses:
                continue # Try generating another hypothesis

            # Create unique key for the hypothesis (set of clauses)
            # Ensure inner lists are tuples for hashability
            hypothesis_key = tuple(sorted(tuple(cl) for cl in hypothesis_clauses))

            if hypothesis_key not in self.hypothesis_database:
                self.hypothesis_database.add(hypothesis_key)
                new_hypotheses.append(Hypothesis(clauses=hypothesis_clauses, complexity=total_complexity))
                # Update clause generation state ONLY when a new hypothesis is added
                self._update_clause_generation_state()
            else:
                 # If duplicate, maybe still advance complexity state to avoid getting stuck?
                 # Or have a separate counter for duplicate hits?
                 # For now, let's advance state even on duplicates to keep exploring.
                 self._update_clause_generation_state()


        if attempts >= max_attempts:
            logger.warning(f"Reached max generation attempts ({max_attempts}), returning {len(new_hypotheses)} hypotheses.")

        return new_hypotheses

    def _generate_random_hypotheses(self):
        """Fallback to original random generation if no history."""
        logger.debug("Generating random hypotheses as fallback.")
        new_hypotheses = []
        input_variables = list(range(1, self.num_inputs * self.input_bits + 1))
        output_variables = list(range(self.num_inputs * self.input_bits + 1,
                                    self.num_inputs * self.input_bits + self.num_outputs + 1))
        attempt_limit = self.new_hypotheses_per_round * 5 # Limit attempts
        attempts = 0

        while len(new_hypotheses) < self.new_hypotheses_per_round and attempts < attempt_limit:
            attempts += 1
            # Generate 1 to 3 random clauses per hypothesis
            num_clauses_for_hyp = random.randint(1, 3)
            hypothesis_clauses = [
                self._generate_random_clause(input_variables, output_variables)
                for _ in range(num_clauses_for_hyp)
            ]
            # Ensure clauses are sorted internally for consistent key generation
            hypothesis_clauses = [sorted(cl) for cl in hypothesis_clauses]

            hypothesis_key = tuple(sorted(tuple(cl) for cl in hypothesis_clauses))
            if hypothesis_key not in self.hypothesis_database:
                self.hypothesis_database.add(hypothesis_key)
                new_hypotheses.append(Hypothesis(clauses=hypothesis_clauses))

            self._update_clause_generation_state() # Still advance complexity state

        return new_hypotheses

    def _generate_random_clause(self, input_variables, output_variables):
        """Original random clause generation logic."""
        input_clause = self._generate_input_clause_random(input_variables)
        output_literal = self._generate_output_literal_random(output_variables)
        # Standard implication form: ~Input OR Output
        return sorted([-lit for lit in input_clause] + [output_literal]) # Sort literals within clause

    def _generate_input_clause_random(self, input_variables):
        effective_size = min(self.current_clause_size, len(input_variables))
        if effective_size <= 0: effective_size = 1 # Ensure size is at least 1
        # Ensure input_variables is not empty
        if not input_variables:
             return []
        combination = random.sample(input_variables, effective_size)
        # Randomly negate literals
        return [var if random.choice([True, False]) else -var for var in combination]

    def _generate_output_literal_random(self, output_variables):
        # Ensure output_variables is not empty
        if not output_variables:
             # This case should ideally not happen if initialized correctly
             logger.error("Cannot generate output literal: output_variables list is empty.")
             return 0 # Return a dummy value or raise error
        output_var = random.choice(output_variables)
        # Randomly negate literal
        return output_var if random.choice([True, False]) else -output_var

    def _update_clause_generation_state(self):
        """Update strategy for exploring clause complexity."""
        # Simple strategy: increment size, wrap around when max is reached
        self.current_clause_size += 1
        if self.current_clause_size > self.max_clause_size:
            # Increase max size more aggressively, but don't exceed absolute max
            # Let's try incrementing max_clause_size by 2 initially, then slow down
            increment = 2 if self.max_clause_size < 6 else 1
            self.max_clause_size = min(self.max_clause_size + increment, self.max_possible_clause_size)
            # Reset to size 2 after hitting max, maybe avoids overly simple clauses?
            self.current_clause_size = 2 if self.max_clause_size > 1 else 1

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
    def __init__(self, num_outputs=5, max_hypotheses=2000, new_hypotheses_per_round=200, init_temp=1.0, beam_width=1000, sls_steps=10):
        # Ensure Algorithm base class is initialized if it has an __init__
        # super().__init__() # Assuming Algorithm is an abstract class or has no __init__
        self.num_outputs = num_outputs
        # max_hypotheses now represents the max number of candidates considered *before* pruning to beam width
        self.max_hypotheses = max_hypotheses
        self.beam_width = beam_width
        self.sls_steps = sls_steps # Number of refinement steps
        self.num_inputs = 9 # Hardcoded for Tic Tac Toe
        self.input_bits = 3 # Hardcoded for Tic Tac Toe
        self.round_count = 0

        self.all_hypotheses = [] # Store all ever generated/kept hypotheses
        self.active_hypotheses = [] # Store currently valid hypotheses
        self.rejected_hypotheses_count = 0

        # History stores tuples: (input_assumptions: List[int], correct_output_literal: int)
        self.history: List[Tuple[List[int], int]] = []
        self.history_set = set() # To quickly check for duplicate observations

        self.cnf_manager = CNFManager(self.num_inputs, self.input_bits, self.num_outputs)
        # base_cnf stores constraints from observations + inherent constraints (one-hot encoding etc)
        self.cnf_manager.initialize_constraints() # Initialize the manager's internal CNF first
        self.base_cnf = self.cnf_manager.cnf      # Assign the initialized CNF object

        self.generator = HypothesisGenerator(self.num_inputs, self.input_bits, self.num_outputs, new_hypotheses_per_round)
        # Validator might not be needed as a separate class if validation logic is in _validate_hypotheses
        # self.validator = HypothesisValidator(temperature=init_temp, target_hypotheses_per_round=new_hypotheses_per_round)
        self.scorer = HypothesisScorer()
        self.converter = BinaryConverter(num_outputs)
        self.stats = HypothesisStats()

        # Attributes for temperature tuning (if desired, needs validator class or integrated logic)
        self.temperature = init_temp
        self.target_hypotheses_per_round = new_hypotheses_per_round
        self.moving_average_hypotheses = float(new_hypotheses_per_round)
        self.alpha_smoothing = 0.1  # Smoothing factor for moving average

    def update_history(self, board_state, guess, correct_label):
        input_binary = self.converter.input_to_binary(board_state)
        output_binary = self.converter.output_to_binary(correct_label)

        # Avoid duplicate observations
        history_key = (input_binary, output_binary)
        if history_key in self.history_set:
            return
        self.history_set.add(history_key)

        logger.debug(f"New observation: input={input_binary}, output={correct_label}")

        # Get input assumptions and correct output literal for the history
        input_constraints_clauses = self.cnf_manager.get_input_constraints(input_binary)
        # Input assumptions are the literals representing the true state of the input
        input_assumptions = [clause[0] for clause in input_constraints_clauses]

        output_constraints_clauses = self.cnf_manager.get_output_constraints(output_binary)
        # Find the positive literal corresponding to the correct output index
        correct_output_var = -1
        for clause in output_constraints_clauses:
             lit = clause[0]
             if lit > 0: # This is the positive literal for the correct output
                 correct_output_var = lit
                 break
        if correct_output_var == -1:
             logger.error(f"Could not determine correct output variable for label {correct_label}, binary {output_binary}")
             return # Cannot proceed without the correct output variable

        # Store the necessary info for validation and generation
        self.history.append((input_assumptions, correct_output_var))

        # Update the base CNF with the new observation
        # Add clauses stating THIS input implies THIS output
        # Form: [-in_lit1, -in_lit2, ..., correct_out_var]
        observation_clause = [-lit for lit in input_assumptions] + [correct_output_var]
        self.base_cnf.append(observation_clause)

        # Add the input constraints themselves to base_cnf to ensure consistency?
        # No, base_cnf should represent the *learned rules/constraints*, not specific instances.
        # The input constraints are passed as *assumptions* during validation/prediction.
        # However, the INITIAL constraints (one-hot encoding) are part of base_cnf.

        logger.debug(f"Total observations: {len(self.history)}")
        logger.debug(f"Total base CNF clauses: {len(self.base_cnf.clauses)}")

        # Update generator with history
        self.generator.set_history(self.history)

        # 1. Generate new hypotheses based on history
        new_hypotheses_guided = self.generator.generate_hypotheses()
        logger.debug(f"Generated {len(new_hypotheses_guided)} new guided hypotheses.")
        for h in new_hypotheses_guided: # Mark as active
             h.is_active = True
             h.last_evaluated = self.round_count
        self.all_hypotheses.extend(new_hypotheses_guided)

        # --- Beam Search Modification Point ---
        # Ensure we mutate *from* the current beam (i.e., top scored active hypotheses)
        # If active_hypotheses is empty (first round), this will do nothing.
        hypotheses_in_beam_to_mutate = self.active_hypotheses # Mutate from the current beam
        num_to_mutate = min(len(hypotheses_in_beam_to_mutate), self.generator.new_hypotheses_per_round // 2)
        # mutated_hypotheses = self._mutate_hypotheses(hypotheses_in_beam_to_mutate, num_to_mutate)
        # --- Replace Mutation with SLS Refinement ---
        refined_hypotheses = self._refine_hypotheses_sls(hypotheses_in_beam_to_mutate, num_to_mutate)
        # --- End Replacement ---

        # 2. Generate hypotheses by mutating existing active ones (now from the beam)
        # Add refined hypotheses instead of mutated ones
        for h in refined_hypotheses:
            h.is_active = True
            h.last_evaluated = self.round_count # Mark as newly generated for scoring bonus
        self.all_hypotheses.extend(refined_hypotheses)

        # 3. Validate ALL potentially active hypotheses (old + guided + refined)
        self._validate_hypotheses() # Updates self.active_hypotheses

        # 4. Prune hypotheses if exceeding max_hypotheses
        self._prune_hypotheses()

        # 5. Score remaining active hypotheses
        self._score_hypotheses()

        # 6. Print stats for this round
        self.stats.round_count = self.round_count
        self.stats.print_stats(self.active_hypotheses)
        self.round_count += 1

        # 7. Auto-tune temperature
        self._auto_tune_temperature()
        print(f"Current temperature: {self.temperature:.2f}")

    # --- Validation Logic (Simplified for Ensemble Role) ---
    def _validate_hypotheses(self):
        # logger.debug("Starting hypothesis validation (SIMPLIFIED - Pass Through)...")
        # In this simplified model, we essentially let most hypotheses pass through.
        # The main validation happens in the external Verifier.
        # We only perform minimal internal checks, like ensuring hypotheses aren't trivially empty or self-contradictory (optional).

        validated_active_hypotheses = []
        rejected_count = 0
        hypotheses_to_check = [h for h in self.all_hypotheses if h.is_active]

        # Keep all hypotheses active for now, external verifier will handle consistency.
        self.active_hypotheses = hypotheses_to_check
        rejected_count = 0 # No internal rejection based on history

        # Optional: Add a check for obviously invalid hypotheses (e.g., empty clauses) if needed
        # Example:
        # final_active = []
        # for h in hypotheses_to_check:
        #     if h.clauses and all(h.clauses):
        #         final_active.append(h)
        #     else:
        #         h.is_active = False
        #         rejected_count += 1
        # self.active_hypotheses = final_active

        logger.debug(f"Internal validation (SIMPLIFIED): Kept {len(self.active_hypotheses)} hypotheses. Rejected {rejected_count} internally.")

    def _prune_hypotheses(self):
        """Prunes active hypotheses list based on beam_width."""
        # First, ensure only currently active hypotheses are considered for pruning
        # (This might be redundant if _validate_hypotheses correctly sets self.active_hypotheses,
        # but provides robustness)
        active_candidates = [h for h in self.all_hypotheses if h.is_active]

        if len(active_candidates) > self.beam_width:
            # Sort by score to determine the beam
            active_candidates.sort(key=lambda h: h.score, reverse=True)
            num_to_remove = len(active_candidates) - self.beam_width
            # Mark hypotheses outside the beam as inactive
            for h in active_candidates[self.beam_width:]:
                h.is_active = False
            # Update active_hypotheses to contain only the beam
            self.active_hypotheses = active_candidates[:self.beam_width]
            logger.debug(f"Pruned {num_to_remove} hypotheses. Keeping beam of {len(self.active_hypotheses)}.")
        else:
             # If within beam width, all active candidates remain
             self.active_hypotheses = active_candidates
             logger.debug(f"Hypothesis count ({len(self.active_hypotheses)}) within beam width ({self.beam_width}). No pruning needed.")

    def _score_hypotheses(self):
        """Scores the currently active hypotheses based on consistency with history."""
        if not self.history:
             # If no history, fallback to complexity scoring
             for h in self.active_hypotheses:
                 complexity = h.complexity if h.complexity > 0 else 1
                 h.score = 1.0 / complexity
             self.active_hypotheses.sort(key=lambda h: h.score, reverse=True)
             return

        logger.debug(f"Scoring {len(self.active_hypotheses)} active hypotheses...")
        # Use a solver for checking consistency efficiently
        # Bootstrapping with base_cnf is NOT correct here, we check hypothesis alone.
        for h in self.active_hypotheses:
            consistent_count = 0
            try:
                 with Solver(bootstrap_with=h.clauses, use_timer=False) as solver:
                     for obs_idx, (input_assumptions, correct_output_lit) in enumerate(self.history):
                         # A hypothesis is consistent with an observation if:
                         # 1. It doesn't force the wrong output (H + I + Not(O) is SAT)
                         # 2. It doesn't allow an incorrect output (H + I + O_wrong is UNSAT)
                         # Let's simplify for scoring: Check if H + I + O is SAT (i.e., consistent)
                         if solver.solve(assumptions=input_assumptions + [correct_output_lit]):
                             consistent_count += 1
                         # A more complex score could also penalize allowing incorrect outputs here.

            except Exception as e:
                 logger.error(f"Error scoring hypothesis {h.clauses}: {e}", exc_info=True)
                 consistent_count = 0 # Penalize hypotheses causing solver errors

            # Score based on fraction of consistent observations + complexity penalty
            consistency_score = consistent_count / len(self.history)
            complexity_penalty = 1.0 / (h.complexity + 1) # Small penalty for complexity
            h.score = consistency_score * complexity_penalty

            # Add a small bonus if it was just generated? (To encourage exploration)
            if h.last_evaluated == self.round_count:
                 h.score += 0.01 # Small novelty bonus

        # Sort by score after updating
        self.active_hypotheses.sort(key=lambda h: h.score, reverse=True)
        # Log top scores
        top_scores = [f"{hyp.score:.4f}" for hyp in self.active_hypotheses[:5]]
        logger.debug(f"Hypothesis scoring complete. Top 5 scores: {top_scores}")

    def _mutate_hypotheses(self, hypotheses_to_mutate: List[Hypothesis], num_mutations: int) -> List[Hypothesis]:
        """DEPRECATED: Replaced by SLS refinement. Kept for reference if needed."""
        mutated_hypotheses = []
        if not hypotheses_to_mutate or num_mutations <= 0:
            return mutated_hypotheses

        for _ in range(num_mutations):
            # Select a hypothesis to mutate (e.g., randomly from active ones)
            parent_hypothesis = random.choice(hypotheses_to_mutate)
            new_clauses = [list(clause) for clause in parent_hypothesis.clauses] # Deep copy

            if not new_clauses: continue # Skip if parent has no clauses

            # Choose a mutation type
            mutation_type = random.choice(['flip', 'add', 'remove', 'change_output'])

            try:
                clause_idx_to_mutate = random.randrange(len(new_clauses))
                clause_to_mutate = new_clauses[clause_idx_to_mutate]

                if mutation_type == 'flip' and clause_to_mutate:
                    lit_idx_to_flip = random.randrange(len(clause_to_mutate))
                    clause_to_mutate[lit_idx_to_flip] *= -1

                elif mutation_type == 'add':
                    # Add a random input literal (avoiding duplicates)
                    input_vars = list(range(1, self.cnf_manager.num_inputs * self.cnf_manager.input_bits + 1))
                    potential_lits = set(input_vars + [-v for v in input_vars])
                    existing_lits = set(clause_to_mutate)
                    available_lits = list(potential_lits - existing_lits)
                    if available_lits:
                        new_lit = random.choice(available_lits)
                        clause_to_mutate.append(new_lit)
                        clause_to_mutate.sort(key=abs) # Keep sorted

                elif mutation_type == 'remove' and len(clause_to_mutate) > 1:
                     # Ensure we don't remove the only literal
                     lit_idx_to_remove = random.randrange(len(clause_to_mutate))
                     del clause_to_mutate[lit_idx_to_remove]

                elif mutation_type == 'change_output' and clause_to_mutate:
                     output_vars = list(range(self.cnf_manager._var_output_bit(0),
                                               self.cnf_manager._var_output_bit(self.num_outputs - 1) + 1))
                     output_lits_in_clause = [l for l in clause_to_mutate if abs(l) in output_vars]
                     if output_lits_in_clause:
                          # Flip the sign of an existing output literal or change it
                          lit_idx = clause_to_mutate.index(random.choice(output_lits_in_clause))
                          clause_to_mutate[lit_idx] *= -1 # Simple flip for now

                # Create new hypothesis if mutation occurred and it's unique
                # Check uniqueness based on clauses
                mutated_key = tuple(sorted(tuple(sorted(cl)) for cl in new_clauses))
                if mutated_key not in self.generator.hypothesis_database: # Use generator's db
                     self.generator.hypothesis_database.add(mutated_key)
                     complexity = sum(len(cl) for cl in new_clauses)
                     mutated_hypotheses.append(Hypothesis(clauses=new_clauses, complexity=complexity))

            except IndexError: # Handle potential empty lists/clauses
                pass
            except Exception as e:
                 logger.error(f"Error during mutation: {e}", exc_info=True)

        logger.debug(f"Generated {len(mutated_hypotheses)} mutated hypotheses.")
        return mutated_hypotheses

    def _refine_hypotheses_sls(self, hypotheses_to_refine: List[Hypothesis], num_to_refine: int) -> List[Hypothesis]:
        """Refines hypotheses using a more targeted Stochastic Local Search."""
        refined_hypotheses = []
        if not hypotheses_to_refine or num_to_refine <= 0 or not self.history:
            logger.debug("Skipping SLS refinement: No hypotheses to refine or no history.")
            return refined_hypotheses

        # Sample hypotheses from the beam to refine
        indices_to_refine = np.random.choice(len(hypotheses_to_refine),
                                             size=min(num_to_refine, len(hypotheses_to_refine)),
                                             replace=False)

        logger.debug(f"Attempting TARGETED SLS refinement on {len(indices_to_refine)} hypotheses for {self.sls_steps} steps each.")

        output_var_start = self.cnf_manager.num_inputs * self.cnf_manager.input_bits + 1
        output_var_end = output_var_start + self.num_outputs

        for i in indices_to_refine:
            parent_hypothesis = hypotheses_to_refine[i]
            current_clauses = [list(clause) for clause in parent_hypothesis.clauses] # Deep copy

            if not current_clauses or not all(clause for clause in current_clauses): # Check for empty clauses too
                # logger.debug(f"Skipping parent {i}: Contains empty clauses.")
                continue # Skip empty or invalid parent

            try:
                # Use a single solver instance for refinement steps of this hypothesis
                with Solver(use_timer=False) as solver:
                    needs_flip = False
                    for step in range(self.sls_steps):
                        needs_flip = False # Reset flip flag for each step
                        # 1. Pick a random observation from history
                        obs_idx = random.randrange(len(self.history))
                        input_assumptions, correct_output_lit = self.history[obs_idx]

                        # 2. Check if current clauses allow the CORRECT output
                        solver.delete()
                        for clause in current_clauses:
                            solver.add_clause(clause)
                        allows_correct = solver.solve(assumptions=input_assumptions + [correct_output_lit])

                        if not allows_correct:
                            # Target 1: Hypothesis doesn't allow the correct output. Need a flip.
                            # logger.debug(f"SLS Step {step} (Parent {i}, Obs {obs_idx}): Doesn't allow correct output {correct_output_lit}. Flipping.")
                            needs_flip = True
                        else:
                            # Target 2: Correct output is allowed. Check if ANY incorrect output is ALSO allowed.
                            for potential_output_var in range(output_var_start, output_var_end):
                                if potential_output_var != correct_output_lit:
                                    incorrect_output_lit = potential_output_var
                                    # Reuse the solver state (clauses are already added)
                                    allows_incorrect = solver.solve(assumptions=input_assumptions + [incorrect_output_lit])
                                    if allows_incorrect:
                                        # logger.debug(f"SLS Step {step} (Parent {i}, Obs {obs_idx}): Allows incorrect output {incorrect_output_lit}. Flipping.")
                                        needs_flip = True
                                        break # Found an ambiguity, no need to check other incorrect outputs

                        # 3. Perform flip if needed
                        if needs_flip:
                            # Heuristic: Pick a random non-empty clause and flip a random literal in it
                            non_empty_clauses = [idx for idx, cl in enumerate(current_clauses) if cl]
                            if not non_empty_clauses:
                                # logger.debug(f"SLS Step {step} (Parent {i}): No non-empty clauses to flip.")
                                break # Cannot flip if no valid clauses

                            clause_idx_to_flip = random.choice(non_empty_clauses)
                            target_clause = current_clauses[clause_idx_to_flip]

                            lit_idx_to_flip = random.randrange(len(target_clause))

                            # Perform the flip
                            target_clause[lit_idx_to_flip] *= -1
                            # Optional: Check if clause becomes empty or tautological? For now, allow.

                # 4. After SLS steps, check uniqueness and add if new
                # Check again for empty clauses created during flips
                if not all(clause for clause in current_clauses):
                   # logger.debug(f"SLS Result (Parent {i}): Contains empty clauses after flips. Discarding.")
                   continue

                final_clauses_sorted = tuple(sorted(tuple(sorted(cl)) for cl in current_clauses))

                if final_clauses_sorted and final_clauses_sorted not in self.generator.hypothesis_database:
                    self.generator.hypothesis_database.add(final_clauses_sorted)
                    complexity = sum(len(cl) for cl in current_clauses)
                    # Convert back to list of lists for Hypothesis object
                    final_clauses_list = [list(c) for c in final_clauses_sorted]
                    refined_hypotheses.append(Hypothesis(clauses=final_clauses_list, complexity=complexity))
                    # logger.debug(f"SLS produced new unique hypothesis from parent {i}")

            except Exception as e:
                logger.error(f"Error during TARGETED SLS refinement for hypothesis from parent {i}: {e}", exc_info=True)

        logger.debug(f"Targeted SLS refinement generated {len(refined_hypotheses)} new unique hypotheses.")
        return refined_hypotheses

    def _auto_tune_temperature(self):
        """Adjusts temperature based on the number of active hypotheses."""
        # This logic might be better placed if HypothesisValidator class is used
        # Simple heuristic adjustment:
        self.moving_average_hypotheses = (1 - self.alpha_smoothing) * self.moving_average_hypotheses + self.alpha_smoothing * len(self.active_hypotheses)

        if self.moving_average_hypotheses > self.target_hypotheses_per_round * 1.1: # Allow some buffer
            self.temperature *= 1.05 # Increase temperature to explore more
        elif self.moving_average_hypotheses < self.target_hypotheses_per_round * 0.9:
            self.temperature *= 0.95 # Decrease temperature to exploit better hypotheses

        self.temperature = max(0.01, min(100, self.temperature)) # Clamp temperature

    def predict(self, board_state):
        """Predicts the output label for a given board state based on active hypotheses."""
        logger.debug(f"Predicting for observation: {board_state}")
        input_binary = self.converter.input_to_binary(board_state)
        input_constraints = self.cnf_manager.get_input_constraints(input_binary)
        input_assumptions = [c[0] for c in input_constraints]

        possible_outputs = []
        output_scores = defaultdict(float)

        if not self.active_hypotheses:
            logger.warning("Predict called with no active hypotheses. Returning default.")
            return 0 # Default prediction (e.g., 'ok')

        # Use a single solver instance for all prediction checks
        try:
            with Solver(bootstrap_with=self.base_cnf.clauses) as solver:
                for output_idx in range(self.num_outputs):
                    output_var = self.cnf_manager._var_output_bit(output_idx)
                    is_possible = False
                    hypothesis_score_sum = 0.0

                    # Check if this output is possible under *any* active hypothesis
                    temp_possible = False
                    for h in self.active_hypotheses:
                        # Check if Input + Hypothesis + This Output is SAT
                        if solver.solve(assumptions = input_assumptions + h.clauses + [output_var]):
                            temp_possible = True
                            hypothesis_score_sum += h.score # Accumulate score of consistent hypotheses

                    if temp_possible:
                        possible_outputs.append(output_idx)
                        output_scores[output_idx] = hypothesis_score_sum

        except Exception as e:
             logger.error(f"Error during prediction solving: {e}", exc_info=True)
             return 0 # Default prediction on error

        if not possible_outputs:
            logger.warning(f"No output possible for input {board_state} according to active hypotheses.")
            return 0 # Default if no output is deemed possible
        elif len(possible_outputs) == 1:
            return possible_outputs[0] # Return the only possible output
        else:
            # If multiple outputs are possible, choose the one supported by highest total score
            best_output = max(possible_outputs, key=lambda idx: output_scores[idx])
            logger.debug(f"Multiple outputs possible ({possible_outputs}), selecting best based on score: {best_output}")
            return best_output

