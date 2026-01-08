"""
SAT-based Few-Shot Rule Learner

This module implements a hypothesis-based learner that:
1. Encodes game states as boolean variables
2. Maintains a set of CNF clauses representing possible rules
3. Eliminates hypotheses inconsistent with observations
4. Predicts based on remaining valid hypotheses

Key Insight: Rules are encoded as implications:
  IF (input_condition) THEN (output_label)
  
In CNF: NOT(input_condition) OR output_label
       = [-input_lit1, -input_lit2, ..., output_lit]
"""

import itertools
from pysat.formula import CNF
from pysat.solvers import Solver
from few_shot_algs.few_shot_alg import Algorithm
import random
import logging
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Hypothesis:
    """Represents a single rule hypothesis as CNF clauses."""
    def __init__(self, clauses: List[List[int]], complexity: int = None):
        self.clauses = clauses
        self.complexity = complexity or sum(len(c) for c in clauses)
        self.score = 0.0
        self.is_active = True
        self.consistent_count = 0
        self.total_applicable = 0
    
    def __repr__(self):
        return f"Hypothesis(clauses={self.clauses}, score={self.score:.3f})"


class VariableEncoder:
    """Encodes TicTacToe states into SAT variables.
    
    Variable Layout (for 9-position, 3-state input + 5 outputs):
    - Positions 1-27: Input state (3 bits per position: Empty/X/O)
      - vars 1,2,3 = position 0 (E, X, O)
      - vars 4,5,6 = position 1
      - ...
      - vars 25,26,27 = position 8
    - Positions 28-32: Output labels (ok, win1, win2, draw, error)
    """
    
    def __init__(self, num_positions=9, states_per_pos=3, num_outputs=5):
        self.num_positions = num_positions
        self.states_per_pos = states_per_pos
        self.num_outputs = num_outputs
        self.num_input_vars = num_positions * states_per_pos
        
        # State mapping: '0'=Empty(index 0), '1'=X(index 1), '2'=O(index 2)
        self.state_map = {'0': 0, '1': 1, '2': 2}
        # Output mapping
        self.output_map = {'ok': 0, 'win1': 1, 'win2': 2, 'draw': 3, 'error': 4}
        self.reverse_output = {v: k for k, v in self.output_map.items()}
    
    def input_var(self, position: int, state_idx: int) -> int:
        """Get variable number for input position/state."""
        return position * self.states_per_pos + state_idx + 1
    
    def output_var(self, label_idx: int) -> int:
        """Get variable number for output label."""
        return self.num_input_vars + label_idx + 1
    
    def encode_input(self, board: str) -> List[int]:
        """Convert board string to list of true variable literals."""
        assumptions = []
        for pos, char in enumerate(board):
            state_idx = self.state_map[char]
            # The variable for this state is true
            true_var = self.input_var(pos, state_idx)
            assumptions.append(true_var)
            # Other states are false
            for other_state in range(self.states_per_pos):
                if other_state != state_idx:
                    assumptions.append(-self.input_var(pos, other_state))
        return assumptions
    
    def encode_output(self, label: str) -> int:
        """Get the positive literal for an output label."""
        return self.output_var(self.output_map[label])
    
    def decode_output(self, var: int) -> Optional[str]:
        """Convert variable to output label if it's an output var."""
        if var > self.num_input_vars:
            idx = var - self.num_input_vars - 1
            return self.reverse_output.get(idx)
        return None


class HypothesisGenerator:
    """Generates rule hypotheses of varying complexity."""
    
    def __init__(self, encoder: VariableEncoder, max_clause_size: int = 3):
        self.encoder = encoder
        self.max_clause_size = max_clause_size
        self.generated = set()  # Track generated hypothesis signatures
    
    def generate_from_observation(self, board: str, label: str, 
                                   num_hypotheses: int = 50) -> List[Hypothesis]:
        """Generate hypotheses consistent with a single observation.
        
        Strategy: Create rules of form "IF (subset of input) THEN output"
        """
        hypotheses = []
        input_assumptions = self.encoder.encode_input(board)
        output_lit = self.encoder.encode_output(label)
        
        # Get the positive input literals (what's actually true)
        true_input_lits = [lit for lit in input_assumptions if lit > 0]
        
        for _ in range(num_hypotheses * 2):  # Generate more, filter dupes
            if len(hypotheses) >= num_hypotheses:
                break
            
            # Pick random subset of input conditions
            subset_size = random.randint(1, min(self.max_clause_size, len(true_input_lits)))
            selected_lits = random.sample(true_input_lits, subset_size)
            
            # Create implication: IF selected_inputs THEN output
            # CNF: OR(-selected_inputs, output)
            clause = [-lit for lit in selected_lits] + [output_lit]
            clause = sorted(clause)
            
            # Check for duplicates
            sig = tuple(clause)
            if sig not in self.generated:
                self.generated.add(sig)
                hypotheses.append(Hypothesis([clause], complexity=len(clause)))
        
        return hypotheses
    
    def generate_general_hypotheses(self, num_hypotheses: int = 100) -> List[Hypothesis]:
        """Generate general hypotheses not tied to specific observations."""
        hypotheses = []
        
        for _ in range(num_hypotheses * 2):
            if len(hypotheses) >= num_hypotheses:
                break
            
            # Random clause size
            input_count = random.randint(1, self.max_clause_size)
            
            # Pick random input positions and states
            positions = random.sample(range(self.encoder.num_positions), 
                                     min(input_count, self.encoder.num_positions))
            
            input_lits = []
            for pos in positions:
                state = random.randint(0, self.encoder.states_per_pos - 1)
                var = self.encoder.input_var(pos, state)
                # Randomly negate (50% chance)
                lit = var if random.random() > 0.5 else -var
                input_lits.append(lit)
            
            # Pick random output
            output_idx = random.randint(0, self.encoder.num_outputs - 1)
            output_lit = self.encoder.output_var(output_idx)
            
            # Create clause
            clause = sorted(input_lits + [output_lit])
            sig = tuple(clause)
            
            if sig not in self.generated:
                self.generated.add(sig)
                hypotheses.append(Hypothesis([clause], complexity=len(clause)))
        
        return hypotheses


class SATRuleLearner(Algorithm):
    """
    Few-shot rule learner using SAT-based hypothesis elimination.
    
    Core Algorithm:
    1. Maintain pool of hypothesis rules (CNF clauses)
    2. On each observation, eliminate hypotheses inconsistent with the data
    3. Predict by voting among remaining valid hypotheses
    """
    
    def __init__(self, num_outputs=5, max_hypotheses=5000, 
                 hypotheses_per_round=100, max_clause_size=4):
        super().__init__()
        self.num_outputs = num_outputs
        self.max_hypotheses = max_hypotheses
        self.hypotheses_per_round = hypotheses_per_round
        
        self.encoder = VariableEncoder(num_outputs=num_outputs)
        self.generator = HypothesisGenerator(self.encoder, max_clause_size)
        
        self.hypotheses: List[Hypothesis] = []
        self.observation_history: List[Tuple[str, str]] = []  # (board, label)
        
        # Statistics
        self.stats = {
            'total_generated': 0,
            'total_eliminated': 0,
            'predictions_made': 0,
        }
    
    def predict(self, observation: str) -> int:
        """Predict label for observation using current hypotheses."""
        self.stats['predictions_made'] += 1
        
        if not self.hypotheses:
            # No hypotheses yet - random guess
            return random.randint(0, self.num_outputs - 1)
        
        input_assumptions = self.encoder.encode_input(observation)
        
        # Count votes for each output from consistent hypotheses
        output_votes = defaultdict(float)
        
        for hyp in self.hypotheses:
            if not hyp.is_active:
                continue
            
            # Check which outputs this hypothesis allows
            for output_idx in range(self.num_outputs):
                output_lit = self.encoder.output_var(output_idx)
                
                # Check if hypothesis + input + output is SAT
                with Solver(bootstrap_with=hyp.clauses) as solver:
                    if solver.solve(assumptions=input_assumptions + [output_lit]):
                        # Weight by hypothesis score/confidence
                        weight = 1.0 + hyp.score
                        output_votes[output_idx] += weight
        
        if not output_votes:
            return random.randint(0, self.num_outputs - 1)
        
        # Return output with highest vote
        return max(output_votes, key=output_votes.get)
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        """Update with new observation, eliminate inconsistent hypotheses."""
        super().update_history(observation, guess, correct_label)
        
        label_str = self.encoder.reverse_output.get(correct_label, 'ok')
        self.observation_history.append((observation, label_str))
        
        # Generate new hypotheses from this observation
        new_hyps = self.generator.generate_from_observation(
            observation, label_str, self.hypotheses_per_round
        )
        self.hypotheses.extend(new_hyps)
        self.stats['total_generated'] += len(new_hyps)
        
        # Eliminate inconsistent hypotheses
        self._eliminate_inconsistent()
        
        # Prune if too many
        if len([h for h in self.hypotheses if h.is_active]) > self.max_hypotheses:
            self._prune_hypotheses()
        
        # Update scores
        self._score_hypotheses()
    
    def _eliminate_inconsistent(self):
        """Remove hypotheses that contradict observed data."""
        eliminated = 0
        
        for hyp in self.hypotheses:
            if not hyp.is_active:
                continue
            
            is_consistent = True
            for board, label in self.observation_history:
                if not self._check_consistency(hyp, board, label):
                    is_consistent = False
                    break
            
            if not is_consistent:
                hyp.is_active = False
                eliminated += 1
        
        self.stats['total_eliminated'] += eliminated
    
    def _check_consistency(self, hyp: Hypothesis, board: str, label: str) -> bool:
        """Check if hypothesis is consistent with observation.
        
        A hypothesis is consistent if:
        - It doesn't FORCE a wrong output (input + hyp + correct_output is SAT)
        """
        input_assumptions = self.encoder.encode_input(board)
        correct_output_lit = self.encoder.encode_output(label)
        
        with Solver(bootstrap_with=hyp.clauses) as solver:
            # Check if hypothesis allows the correct output
            if solver.solve(assumptions=input_assumptions + [correct_output_lit]):
                return True
            return False
    
    def _score_hypotheses(self):
        """Score hypotheses by consistency with history."""
        for hyp in self.hypotheses:
            if not hyp.is_active:
                continue
            
            consistent = 0
            applicable = 0
            
            for board, label in self.observation_history:
                input_assumptions = self.encoder.encode_input(board)
                correct_output_lit = self.encoder.encode_output(label)
                
                with Solver(bootstrap_with=hyp.clauses) as solver:
                    # Check if rule is applicable (input matches)
                    if solver.solve(assumptions=input_assumptions):
                        applicable += 1
                        if solver.solve(assumptions=input_assumptions + [correct_output_lit]):
                            consistent += 1
            
            hyp.consistent_count = consistent
            hyp.total_applicable = applicable
            
            if applicable > 0:
                hyp.score = consistent / applicable
            else:
                hyp.score = 0.5  # Neutral score for non-applicable rules
    
    def _prune_hypotheses(self):
        """Keep only the best hypotheses."""
        active = [h for h in self.hypotheses if h.is_active]
        # Sort by score (desc), then complexity (asc)
        active.sort(key=lambda h: (-h.score, h.complexity))
        
        # Keep top hypotheses
        keep = set(id(h) for h in active[:self.max_hypotheses])
        
        for h in self.hypotheses:
            if id(h) not in keep:
                h.is_active = False
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        active_count = sum(1 for h in self.hypotheses if h.is_active)
        return {
            **self.stats,
            'active_hypotheses': active_count,
            'observations': len(self.observation_history),
        }
    
    def get_top_hypotheses(self, n: int = 10) -> List[Hypothesis]:
        """Get top-scoring active hypotheses."""
        active = [h for h in self.hypotheses if h.is_active]
        active.sort(key=lambda h: (-h.score, h.complexity))
        return active[:n]
    
    def explain_hypothesis(self, hyp: Hypothesis) -> str:
        """Convert hypothesis to human-readable form."""
        parts = []
        for clause in hyp.clauses:
            input_conds = []
            output = None
            
            for lit in clause:
                var = abs(lit)
                is_neg = lit < 0
                
                if var <= self.encoder.num_input_vars:
                    # Input variable
                    pos = (var - 1) // self.encoder.states_per_pos
                    state = (var - 1) % self.encoder.states_per_pos
                    state_char = {0: 'E', 1: 'X', 2: 'O'}[state]
                    
                    if is_neg:
                        input_conds.append(f"pos{pos}={state_char}")
                    else:
                        input_conds.append(f"pos{pos}!={state_char}")
                else:
                    # Output variable
                    idx = var - self.encoder.num_input_vars - 1
                    label = self.encoder.reverse_output.get(idx, '?')
                    if not is_neg:
                        output = label
            
            if input_conds and output:
                parts.append(f"IF {' AND '.join(input_conds)} THEN {output}")
            elif output:
                parts.append(f"ALWAYS {output}")
        
        return "; ".join(parts) if parts else str(hyp.clauses)


# Wrapper for compatibility with existing test framework
class SATHypothesesAlgorithm(SATRuleLearner):
    """Alias for backward compatibility."""
    pass


if __name__ == "__main__":
    # Quick test
    from tictactoe import tictactoe, random_board, label_space
    
    learner = SATRuleLearner(num_outputs=5)
    
    correct = 0
    total = 100
    
    for i in range(total):
        board = random_board()
        true_label = tictactoe(board)
        true_idx = label_space.index(true_label)
        
        pred_idx = learner.predict(board)
        
        if pred_idx == true_idx:
            correct += 1
        
        learner.update_history(board, pred_idx, true_idx)
        
        if (i + 1) % 20 == 0:
            stats = learner.get_stats()
            print(f"Round {i+1}: Acc={correct/(i+1):.2%}, Active={stats['active_hypotheses']}")
    
    print(f"\nFinal accuracy: {correct/total:.2%}")
    print(f"Stats: {learner.get_stats()}")
    
    print("\nTop hypotheses:")
    for hyp in learner.get_top_hypotheses(5):
        print(f"  Score={hyp.score:.3f}: {learner.explain_hypothesis(hyp)}")
