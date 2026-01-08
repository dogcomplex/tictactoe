"""
Rule-based Few-Shot Learner (No SAT dependency)

This implements a hypothesis-based rule learner using pure Python.
Instead of SAT solving, we use direct pattern matching and constraint propagation.

Rules are encoded as:
  Pattern -> Label
  
Where Pattern is a partial specification of board positions.
"""

import random
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A rule that maps input patterns to output labels.
    
    Pattern format: dict mapping position -> required_state
    - position: 0-8 for board positions
    - required_state: '0', '1', '2', or None (don't care)
    """
    pattern: Dict[int, str]  # position -> required state
    output: int  # output label index
    confidence: float = 0.5
    correct_count: int = 0
    applicable_count: int = 0
    
    def matches(self, board: str) -> bool:
        """Check if board matches this rule's pattern."""
        for pos, required_state in self.pattern.items():
            if board[pos] != required_state:
                return False
        return True
    
    def complexity(self) -> int:
        """Return rule complexity (number of conditions)."""
        return len(self.pattern)
    
    def signature(self) -> tuple:
        """Return hashable signature for deduplication."""
        return (tuple(sorted(self.pattern.items())), self.output)
    
    def __repr__(self):
        conds = [f"p{k}={v}" for k, v in sorted(self.pattern.items())]
        return f"Rule({' AND '.join(conds)} -> {self.output}, conf={self.confidence:.2f})"


class RuleGenerator:
    """Generates candidate rules from observations."""
    
    def __init__(self, num_positions=9, num_outputs=5, max_conditions=4):
        self.num_positions = num_positions
        self.num_outputs = num_outputs
        self.max_conditions = max_conditions
        self.generated_signatures: Set[tuple] = set()
    
    def from_observation(self, board: str, label: int, 
                        count: int = 30) -> List[Rule]:
        """Generate rules consistent with a single observation."""
        rules = []
        
        for _ in range(count * 2):  # Generate more, filter dupes
            if len(rules) >= count:
                break
            
            # Pick random subset of positions to include in pattern
            num_conds = random.randint(1, min(self.max_conditions, self.num_positions))
            positions = random.sample(range(self.num_positions), num_conds)
            
            # Create pattern from actual board state
            pattern = {pos: board[pos] for pos in positions}
            
            rule = Rule(pattern=pattern, output=label)
            sig = rule.signature()
            
            if sig not in self.generated_signatures:
                self.generated_signatures.add(sig)
                rules.append(rule)
        
        return rules
    
    def generate_random(self, count: int = 50) -> List[Rule]:
        """Generate random rules (for exploration)."""
        rules = []
        states = ['0', '1', '2']
        
        for _ in range(count * 2):
            if len(rules) >= count:
                break
            
            num_conds = random.randint(1, self.max_conditions)
            positions = random.sample(range(self.num_positions), num_conds)
            
            pattern = {pos: random.choice(states) for pos in positions}
            output = random.randint(0, self.num_outputs - 1)
            
            rule = Rule(pattern=pattern, output=output)
            sig = rule.signature()
            
            if sig not in self.generated_signatures:
                self.generated_signatures.add(sig)
                rules.append(rule)
        
        return rules


class RuleLearner:
    """
    Few-shot rule learner using hypothesis elimination.
    
    Algorithm:
    1. Generate candidate rules from observations
    2. Eliminate rules that contradict observations
    3. Predict by weighted voting among matching rules
    """
    
    def __init__(self, num_outputs=5, max_rules=2000, 
                 rules_per_round=50, max_conditions=4):
        self.num_outputs = num_outputs
        self.max_rules = max_rules
        self.rules_per_round = rules_per_round
        
        self.generator = RuleGenerator(
            num_outputs=num_outputs, 
            max_conditions=max_conditions
        )
        
        self.rules: List[Rule] = []
        self.observations: List[Tuple[str, int]] = []  # (board, label)
        self.history = []  # For Algorithm interface
        
        self.stats = {
            'total_generated': 0,
            'total_eliminated': 0,
            'predictions': 0,
        }
    
    def predict(self, observation: str) -> int:
        """Predict label using weighted voting among matching rules."""
        self.stats['predictions'] += 1
        
        if not self.rules:
            return random.randint(0, self.num_outputs - 1)
        
        # Find all rules that match this input
        matching_rules = [r for r in self.rules if r.matches(observation)]
        
        if not matching_rules:
            # No matching rules - use all rules weighted by confidence
            return self._fallback_predict()
        
        # Weighted voting
        votes = defaultdict(float)
        for rule in matching_rules:
            weight = rule.confidence * (1.0 / rule.complexity())  # Favor simpler rules
            votes[rule.output] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        return random.randint(0, self.num_outputs - 1)
    
    def _fallback_predict(self) -> int:
        """Fallback prediction when no rules match."""
        # Use prior distribution from observations
        if not self.observations:
            return random.randint(0, self.num_outputs - 1)
        
        label_counts = defaultdict(int)
        for _, label in self.observations:
            label_counts[label] += 1
        
        total = sum(label_counts.values())
        if total == 0:
            return random.randint(0, self.num_outputs - 1)
        
        # Weighted random choice
        rand_val = random.random() * total
        cumsum = 0
        for label, count in label_counts.items():
            cumsum += count
            if rand_val <= cumsum:
                return label
        
        return max(label_counts, key=label_counts.get)
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        """Update with new observation."""
        self.history.append((observation, guess, correct_label))
        self.observations.append((observation, correct_label))
        
        # Generate new rules from this observation
        new_rules = self.generator.from_observation(
            observation, correct_label, self.rules_per_round
        )
        self.rules.extend(new_rules)
        self.stats['total_generated'] += len(new_rules)
        
        # Eliminate contradicted rules
        self._eliminate_contradicted()
        
        # Update rule confidences
        self._update_confidences()
        
        # Prune if too many rules
        if len(self.rules) > self.max_rules:
            self._prune_rules()
    
    def _eliminate_contradicted(self):
        """Remove rules that contradict observations."""
        valid_rules = []
        eliminated = 0
        
        for rule in self.rules:
            is_valid = True
            for board, label in self.observations:
                if rule.matches(board) and rule.output != label:
                    # Rule predicts wrong output for this input
                    is_valid = False
                    break
            
            if is_valid:
                valid_rules.append(rule)
            else:
                eliminated += 1
        
        self.rules = valid_rules
        self.stats['total_eliminated'] += eliminated
    
    def _update_confidences(self):
        """Update rule confidences based on observation history."""
        for rule in self.rules:
            correct = 0
            applicable = 0
            
            for board, label in self.observations:
                if rule.matches(board):
                    applicable += 1
                    if rule.output == label:
                        correct += 1
            
            rule.applicable_count = applicable
            rule.correct_count = correct
            
            if applicable > 0:
                rule.confidence = correct / applicable
            else:
                rule.confidence = 0.5  # Neutral for non-applicable rules
    
    def _prune_rules(self):
        """Keep only the best rules."""
        # Sort by confidence (desc), then complexity (asc, simpler is better)
        self.rules.sort(key=lambda r: (-r.confidence, r.complexity()))
        self.rules = self.rules[:self.max_rules]
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            **self.stats,
            'active_rules': len(self.rules),
            'observations': len(self.observations),
        }
    
    def get_top_rules(self, n: int = 10) -> List[Rule]:
        """Get top rules by confidence."""
        sorted_rules = sorted(self.rules, key=lambda r: (-r.confidence, r.complexity()))
        return sorted_rules[:n]


# Pattern-based learner with more sophisticated rule structures
class PatternLearner(RuleLearner):
    """
    Extended learner that can discover structural patterns like:
    - Row/column/diagonal win conditions
    - Complex boolean combinations
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.discovered_patterns: Dict[str, Rule] = {}  # Named patterns
    
    def _discover_structural_patterns(self):
        """Try to discover row/column/diagonal patterns."""
        # Check for row patterns
        rows = [[0,1,2], [3,4,5], [6,7,8]]
        cols = [[0,3,6], [1,4,7], [2,5,8]]
        diags = [[0,4,8], [2,4,6]]
        
        for pattern_type, positions_list in [('row', rows), ('col', cols), ('diag', diags)]:
            for positions in positions_list:
                # Check if we have observations consistent with this being a win condition
                for player in ['1', '2']:
                    win_label = 1 if player == '1' else 2  # win1 or win2
                    
                    pattern = {pos: player for pos in positions}
                    test_rule = Rule(pattern=pattern, output=win_label)
                    
                    # Check consistency with all observations
                    consistent = True
                    applicable = 0
                    correct = 0
                    
                    for board, label in self.observations:
                        if test_rule.matches(board):
                            applicable += 1
                            if label == win_label:
                                correct += 1
                            else:
                                # This pattern exists but doesn't produce expected win
                                consistent = False
                                break
                    
                    if consistent and applicable > 0:
                        name = f"{pattern_type}_{player}_{''.join(map(str, positions))}"
                        if name not in self.discovered_patterns:
                            test_rule.confidence = correct / applicable if applicable > 0 else 0.5
                            self.discovered_patterns[name] = test_rule
                            logger.info(f"Discovered pattern: {name} -> {test_rule}")
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        """Update and try to discover patterns."""
        super().update_history(observation, guess, correct_label)
        
        # Periodically try to discover structural patterns
        if len(self.observations) % 10 == 0:
            self._discover_structural_patterns()


# Alias for compatibility
class Algorithm:
    """Base algorithm interface."""
    def __init__(self):
        self.history = []
    
    def predict(self, observation: str) -> int:
        raise NotImplementedError
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        self.history.append((observation, guess, correct_label))


class SATHypothesesAlgorithm(PatternLearner, Algorithm):
    """Compatibility alias."""
    pass


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from tictactoe import tictactoe, random_board, label_space
    
    print("Testing RuleLearner...")
    learner = RuleLearner(num_outputs=5, max_rules=1000, rules_per_round=30)
    
    correct = 0
    total = 200
    
    for i in range(total):
        board = random_board()
        true_label = tictactoe(board)
        true_idx = label_space.index(true_label)
        
        pred_idx = learner.predict(board)
        
        if pred_idx == true_idx:
            correct += 1
        
        learner.update_history(board, pred_idx, true_idx)
        
        if (i + 1) % 40 == 0:
            stats = learner.get_stats()
            print(f"Round {i+1}: Acc={correct/(i+1):.1%}, Rules={stats['active_rules']}, "
                  f"Gen={stats['total_generated']}, Elim={stats['total_eliminated']}")
    
    print(f"\nFinal accuracy: {correct/total:.1%}")
    print(f"Final stats: {learner.get_stats()}")
    
    print("\nTop 10 rules:")
    for rule in learner.get_top_rules(10):
        print(f"  {rule}")
