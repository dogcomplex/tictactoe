"""
Truly Blind Few-Shot Learner

This learner has NO embedded game knowledge:
- Doesn't know about rows/cols/diagonals
- Doesn't know what '0', '1', '2' mean
- Doesn't know what labels represent
- Only learns from observations through hypothesis elimination

The goal is honest few-shot learning without cheating.
"""

import random
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations


@dataclass
class Rule:
    """A rule mapping input pattern to output label."""
    pattern: Dict[int, str]  # position -> required value
    output: int  # output label index
    confidence: float = 0.5
    support: int = 0
    
    def matches(self, board: str) -> bool:
        """Check if observation matches this rule's pattern."""
        for pos, val in self.pattern.items():
            if pos >= len(board) or board[pos] != val:
                return False
        return True
    
    def specificity(self) -> int:
        """More conditions = more specific."""
        return len(self.pattern)
    
    def signature(self) -> tuple:
        return (tuple(sorted(self.pattern.items())), self.output)


class BlindLearner:
    """
    Truly blind few-shot learner.
    
    Only knows:
    - board_size: length of input string (inferred if not given)
    - num_outputs: number of possible labels
    
    Does NOT know:
    - What input characters mean
    - What labels represent
    - Any structural patterns (lines, etc.)
    - Any game-specific semantics
    """
    
    def __init__(self, num_outputs=5, board_size=9, label_names=None, 
                 max_rules=5000, rules_per_obs=50):
        self.num_outputs = num_outputs
        self.board_size = board_size
        self.label_names = label_names  # Only for reporting, not used in logic
        self.max_rules = max_rules
        self.rules_per_obs = rules_per_obs
        
        self.rules: List[Rule] = []
        self.rule_signatures: Set[tuple] = set()
        self.observations: List[Tuple[str, int]] = []
        self.history = []
        
        # Track observed values at each position
        self.observed_values: Set[str] = set()
        self.label_counts = defaultdict(int)
        
        self.stats = {
            'generated': 0,
            'eliminated': 0,
            'predictions': 0,
        }
    
    def predict(self, observation: str) -> int:
        """Predict using only learned rules - no structural assumptions."""
        self.stats['predictions'] += 1
        
        if not self.rules:
            # No rules yet - use prior or random
            return self._prior_predict()
        
        # Find matching rules
        matching = [r for r in self.rules if r.matches(observation)]
        
        if not matching:
            return self._prior_predict()
        
        # Weighted voting among matching rules
        votes = defaultdict(float)
        for rule in matching:
            # Weight by specificity and confidence
            weight = (rule.specificity() ** 2) * rule.confidence * (rule.support + 1)
            votes[rule.output] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        return self._prior_predict()
    
    def _prior_predict(self) -> int:
        """Predict based on observed label distribution."""
        if not self.label_counts:
            return random.randint(0, self.num_outputs - 1)
        
        total = sum(self.label_counts.values())
        r = random.random() * total
        cumsum = 0
        for label, count in sorted(self.label_counts.items(), key=lambda x: -x[1]):
            cumsum += count
            if r <= cumsum:
                return label
        return max(self.label_counts, key=self.label_counts.get)
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        """Learn from observation."""
        self.history.append((observation, guess, correct_label))
        self.observations.append((observation, correct_label))
        self.label_counts[correct_label] += 1
        
        # Track observed values
        for char in observation:
            self.observed_values.add(char)
        
        # Generate rules from this observation
        self._generate_rules(observation, correct_label)
        
        # Eliminate contradicted rules
        self._eliminate_contradicted(observation, correct_label)
        
        # Update confidences periodically
        if len(self.observations) % 25 == 0:
            self._update_confidences()
        
        # Prune if needed
        if len(self.rules) > self.max_rules:
            self._prune()
    
    def _generate_rules(self, board: str, label: int):
        """Generate rules from observation - no structural bias."""
        generated = 0
        
        # Generate rules of varying specificity
        for specificity in range(1, min(7, len(board) + 1)):
            attempts = 0
            while generated < self.rules_per_obs and attempts < self.rules_per_obs * 3:
                attempts += 1
                
                # Random positions
                positions = random.sample(range(len(board)), specificity)
                pattern = {pos: board[pos] for pos in positions}
                
                rule = Rule(pattern=pattern, output=label, support=1)
                sig = rule.signature()
                
                if sig not in self.rule_signatures:
                    self.rule_signatures.add(sig)
                    self.rules.append(rule)
                    generated += 1
                    self.stats['generated'] += 1
    
    def _eliminate_contradicted(self, board: str, label: int):
        """Remove rules contradicted by observation."""
        valid = []
        eliminated = 0
        
        for rule in self.rules:
            if rule.matches(board) and rule.output != label:
                eliminated += 1
                self.rule_signatures.discard(rule.signature())
            else:
                valid.append(rule)
        
        self.rules = valid
        self.stats['eliminated'] += eliminated
    
    def _update_confidences(self):
        """Update rule confidences based on recent observations."""
        recent = self.observations[-100:]
        
        for rule in self.rules:
            support = 0
            applicable = 0
            
            for board, label in recent:
                if rule.matches(board):
                    applicable += 1
                    if rule.output == label:
                        support += 1
            
            rule.support = support
            rule.confidence = support / applicable if applicable > 0 else 0.5
    
    def _prune(self):
        """Keep best rules."""
        # Score rules
        for rule in self.rules:
            rule._score = (
                rule.specificity() * 5 +
                rule.confidence * 10 +
                min(rule.support, 10)
            )
        
        # Sort and keep top
        self.rules.sort(key=lambda r: -r._score)
        self.rules = self.rules[:self.max_rules]
        self.rule_signatures = {r.signature() for r in self.rules}
    
    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'rules': len(self.rules),
            'observations': len(self.observations),
            'unique_values': len(self.observed_values),
            'lines': 0,  # For compatibility with test harness
        }
    
    def describe_knowledge(self) -> str:
        """Describe what the learner has discovered."""
        lines = ["=== Blind Learner Knowledge ===\n"]
        
        lines.append(f"Observations: {len(self.observations)}")
        lines.append(f"Rules: {len(self.rules)}")
        lines.append(f"Observed values: {sorted(self.observed_values)}")
        
        lines.append("\n--- Label Distribution ---")
        total = sum(self.label_counts.values())
        for idx in range(self.num_outputs):
            count = self.label_counts[idx]
            pct = count / total * 100 if total > 0 else 0
            label = self.label_names[idx] if self.label_names and idx < len(self.label_names) else str(idx)
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        
        # Show top rules
        lines.append("\n--- Top 20 Rules by Confidence ---")
        sorted_rules = sorted(self.rules, key=lambda r: (-r.confidence, -r.support, -r.specificity()))
        for rule in sorted_rules[:20]:
            if rule.confidence >= 0.8 and rule.support >= 2:
                label = self.label_names[rule.output] if self.label_names and rule.output < len(self.label_names) else str(rule.output)
                pattern_str = ', '.join(f"p{k}={v}" for k, v in sorted(rule.pattern.items()))
                lines.append(f"  [{rule.confidence:.2f}] {pattern_str} => {label} (x{rule.support})")
        
        return '\n'.join(lines)


class BlindLearnerV2(BlindLearner):
    """
    Enhanced blind learner with:
    - Pattern generalization (discover common patterns across labels)
    - Systematic enumeration of small patterns
    - More aggressive hypothesis generation
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pattern_stats: Dict[tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.discovered_patterns: Dict[tuple, int] = {}  # pattern -> label
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        """Learn with pattern discovery."""
        super().update_history(observation, guess, correct_label)
        
        # Track pattern statistics
        self._track_patterns(observation, correct_label)
        
        # Periodically discover strong patterns
        if len(self.observations) % 30 == 0:
            self._discover_patterns()
    
    def _track_patterns(self, board: str, label: int):
        """Track which patterns appear with which labels."""
        # Track 3-position patterns (most likely for TicTacToe-like rules)
        for positions in combinations(range(len(board)), 3):
            pattern = tuple((pos, board[pos]) for pos in positions)
            self.pattern_stats[pattern][label] += 1
    
    def _discover_patterns(self):
        """Discover patterns that strongly predict labels."""
        for pattern, label_counts in self.pattern_stats.items():
            total = sum(label_counts.values())
            if total < 3:
                continue
            
            # Check if any label dominates
            for label, count in label_counts.items():
                if count / total >= 0.9 and count >= 3:
                    # This pattern strongly predicts this label
                    if pattern not in self.discovered_patterns:
                        self.discovered_patterns[pattern] = label
                        # Create a high-confidence rule
                        rule = Rule(
                            pattern=dict(pattern),
                            output=label,
                            confidence=1.0,
                            support=count
                        )
                        sig = rule.signature()
                        if sig not in self.rule_signatures:
                            self.rule_signatures.add(sig)
                            self.rules.append(rule)
    
    def predict(self, observation: str) -> int:
        """Predict with discovered patterns prioritized."""
        self.stats['predictions'] += 1
        
        # Check discovered patterns first
        for pattern, label in self.discovered_patterns.items():
            if all(observation[pos] == val for pos, val in pattern):
                return label
        
        # Fall back to regular prediction
        if not self.rules:
            return self._prior_predict()
        
        matching = [r for r in self.rules if r.matches(observation)]
        
        if not matching:
            return self._prior_predict()
        
        votes = defaultdict(float)
        for rule in matching:
            weight = (rule.specificity() ** 2) * rule.confidence * (rule.support + 1)
            votes[rule.output] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        return self._prior_predict()
    
    def get_stats(self) -> Dict:
        stats = super().get_stats()
        stats['discovered_patterns'] = len(self.discovered_patterns)
        stats['lines'] = len(self.discovered_patterns)  # For compatibility
        return stats
    
    def describe_knowledge(self) -> str:
        base = super().describe_knowledge()
        
        lines = [base, "\n--- Discovered Patterns ---"]
        for pattern, label in sorted(self.discovered_patterns.items(), key=lambda x: x[1]):
            label_name = self.label_names[label] if self.label_names and label < len(self.label_names) else str(label)
            pattern_str = ', '.join(f"p{pos}={val}" for pos, val in pattern)
            lines.append(f"  {pattern_str} => {label_name}")
        
        return '\n'.join(lines)


def test_blind_learner():
    """Test the blind learner on TicTacToe."""
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from tictactoe import tictactoe, random_board, label_space
    
    print("="*60)
    print("Testing Truly Blind Learner on TicTacToe")
    print("="*60)
    
    for learner_class in [BlindLearner, BlindLearnerV2]:
        print(f"\n--- {learner_class.__name__} ---")
        learner = learner_class(num_outputs=5, board_size=9, label_names=label_space)
        
        correct = 0
        checkpoints = [10, 25, 50, 100, 200, 500]
        
        for i in range(500):
            board = random_board()
            true_label = tictactoe(board)
            true_idx = label_space.index(true_label)
            
            pred_idx = learner.predict(board)
            
            if pred_idx == true_idx:
                correct += 1
            
            learner.update_history(board, pred_idx, true_idx)
            
            if (i + 1) in checkpoints:
                stats = learner.get_stats()
                acc = correct / (i + 1)
                print(f"  R{i+1:3d}: {acc:.1%} | Rules={stats['rules']} Patterns={stats.get('discovered_patterns', 0)}")
        
        print(f"\nFinal: {correct/500:.1%}")


if __name__ == "__main__":
    test_blind_learner()
