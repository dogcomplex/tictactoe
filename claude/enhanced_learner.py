"""
Enhanced Few-Shot Rule Learner

Key improvements over basic version:
1. Priority-based rule matching (specific before general)
2. Active pattern discovery for structural rules
3. Better handling of rare labels (wins, draws)
4. Rule refinement through counterexample analysis
"""

import random
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A rule mapping input pattern to output label."""
    pattern: Dict[int, str]  # position -> required state
    output: int  # output label index
    confidence: float = 0.5
    support: int = 0  # How many observations support this rule
    counter_examples: int = 0  # How many contradict it
    priority: int = 0  # Higher = checked first
    
    def matches(self, board: str) -> bool:
        """Check if board matches this rule's pattern."""
        for pos, state in self.pattern.items():
            if board[pos] != state:
                return False
        return True
    
    def specificity(self) -> int:
        """More conditions = more specific."""
        return len(self.pattern)
    
    def signature(self) -> tuple:
        return (tuple(sorted(self.pattern.items())), self.output)


class EnhancedRuleLearner:
    """
    Improved few-shot learner with structural pattern discovery.
    """
    
    # Known structural patterns for TicTacToe-like games
    ROWS = [[0,1,2], [3,4,5], [6,7,8]]
    COLS = [[0,3,6], [1,4,7], [2,5,8]]
    DIAGS = [[0,4,8], [2,4,6]]
    
    def __init__(self, num_outputs=5, max_rules=3000, label_names=None):
        self.num_outputs = num_outputs
        self.max_rules = max_rules
        self.label_names = label_names or ['ok', 'win1', 'win2', 'draw', 'error']
        
        self.rules: List[Rule] = []
        self.rule_signatures: Set[tuple] = set()
        self.observations: List[Tuple[str, int]] = []
        self.history = []  # Algorithm interface
        
        # Label-specific rules (for faster lookup)
        self.rules_by_output: Dict[int, List[Rule]] = defaultdict(list)
        
        # Discovered structural patterns
        self.structural_rules: Dict[str, Rule] = {}
        
        # Prior distribution
        self.label_counts = defaultdict(int)
        
        self.stats = {
            'generated': 0,
            'eliminated': 0,
            'predictions': 0,
            'structural_discovered': 0,
        }
    
    def predict(self, board: str) -> int:
        """Predict using priority-ordered rule matching."""
        self.stats['predictions'] += 1
        
        # 1. Check structural rules first (highest priority)
        for name, rule in self.structural_rules.items():
            if rule.matches(board):
                return rule.output
        
        # 2. Find all matching rules
        matching = []
        for rule in self.rules:
            if rule.matches(board):
                matching.append(rule)
        
        if not matching:
            return self._prior_predict()
        
        # 3. Sort by specificity (more specific = better), then confidence
        matching.sort(key=lambda r: (-r.specificity(), -r.confidence, -r.support))
        
        # 4. Weighted voting among top matches
        votes = defaultdict(float)
        for rule in matching[:20]:  # Consider top 20
            weight = (rule.specificity() + 1) * rule.confidence * (rule.support + 1)
            votes[rule.output] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        return matching[0].output
    
    def _prior_predict(self) -> int:
        """Predict based on label distribution."""
        if not self.label_counts:
            return 0  # Default to 'ok'
        
        total = sum(self.label_counts.values())
        r = random.random() * total
        cumsum = 0
        for label, count in self.label_counts.items():
            cumsum += count
            if r <= cumsum:
                return label
        return max(self.label_counts, key=self.label_counts.get)
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        """Learn from new observation."""
        self.history.append((observation, guess, correct_label))
        self.observations.append((observation, correct_label))
        self.label_counts[correct_label] += 1
        
        # Generate rules from this observation
        self._generate_rules(observation, correct_label)
        
        # Try structural pattern discovery
        self._discover_patterns(observation, correct_label)
        
        # Eliminate contradicted rules
        self._eliminate_contradicted(observation, correct_label)
        
        # Update confidences
        self._update_confidences()
        
        # Prune if needed
        if len(self.rules) > self.max_rules:
            self._prune()
    
    def _generate_rules(self, board: str, label: int, count: int = 40):
        """Generate rules from observation."""
        generated = 0
        
        # Generate rules of varying specificity
        for specificity in range(1, 6):
            for _ in range(count // 5):
                if generated >= count:
                    return
                
                # Random positions
                positions = random.sample(range(9), min(specificity, 9))
                pattern = {pos: board[pos] for pos in positions}
                
                rule = Rule(pattern=pattern, output=label, support=1)
                sig = rule.signature()
                
                if sig not in self.rule_signatures:
                    self.rule_signatures.add(sig)
                    self.rules.append(rule)
                    self.rules_by_output[label].append(rule)
                    generated += 1
                    self.stats['generated'] += 1
    
    def _discover_patterns(self, board: str, label: int):
        """Try to discover structural patterns like win conditions."""
        if label not in [1, 2]:  # Only for wins
            return
        
        player = '1' if label == 1 else '2'
        
        # Check all line patterns
        all_lines = self.ROWS + self.COLS + self.DIAGS
        
        for line in all_lines:
            # Check if this line has all same player
            if all(board[pos] == player for pos in line):
                pattern = {pos: player for pos in line}
                name = f"line_{player}_{''.join(map(str, line))}"
                
                if name not in self.structural_rules:
                    rule = Rule(pattern=pattern, output=label, 
                               confidence=1.0, support=1, priority=100)
                    
                    # Verify against history
                    contradicted = False
                    for obs_board, obs_label in self.observations[:-1]:
                        if rule.matches(obs_board) and obs_label != label:
                            contradicted = True
                            break
                    
                    if not contradicted:
                        self.structural_rules[name] = rule
                        self.stats['structural_discovered'] += 1
                        logger.info(f"Discovered: {name} -> {self.label_names[label]}")
                else:
                    # Update support
                    self.structural_rules[name].support += 1
    
    def _eliminate_contradicted(self, board: str, label: int):
        """Remove rules contradicted by this observation."""
        valid_rules = []
        eliminated = 0
        
        for rule in self.rules:
            if rule.matches(board) and rule.output != label:
                eliminated += 1
            else:
                valid_rules.append(rule)
        
        self.rules = valid_rules
        self.stats['eliminated'] += eliminated
        
        # Rebuild rules_by_output
        self.rules_by_output.clear()
        for rule in self.rules:
            self.rules_by_output[rule.output].append(rule)
        
        # Also eliminate invalid structural rules
        invalid = []
        for name, rule in self.structural_rules.items():
            if rule.matches(board) and rule.output != label:
                invalid.append(name)
        for name in invalid:
            del self.structural_rules[name]
    
    def _update_confidences(self):
        """Update rule confidences."""
        for rule in self.rules:
            support = 0
            applicable = 0
            
            for board, label in self.observations[-100:]:  # Recent history
                if rule.matches(board):
                    applicable += 1
                    if rule.output == label:
                        support += 1
            
            rule.support = support
            if applicable > 0:
                rule.confidence = support / applicable
    
    def _prune(self):
        """Keep best rules."""
        # Score rules
        for rule in self.rules:
            rule.priority = (
                rule.specificity() * 10 + 
                rule.confidence * 5 + 
                min(rule.support, 10)
            )
        
        # Sort and keep top
        self.rules.sort(key=lambda r: -r.priority)
        self.rules = self.rules[:self.max_rules]
        
        # Update signatures
        self.rule_signatures = {r.signature() for r in self.rules}
        
        # Rebuild by-output index
        self.rules_by_output.clear()
        for rule in self.rules:
            self.rules_by_output[rule.output].append(rule)
    
    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'rules': len(self.rules),
            'structural': len(self.structural_rules),
            'observations': len(self.observations),
        }
    
    def get_top_rules(self, n: int = 10) -> List[Rule]:
        sorted_rules = sorted(self.rules, 
                             key=lambda r: (-r.confidence, -r.specificity(), -r.support))
        return sorted_rules[:n]
    
    def get_structural_rules(self) -> Dict[str, Rule]:
        return self.structural_rules.copy()


def run_test(learner_class, rounds=300, verbose=True):
    """Test a learner on TicTacToe."""
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from tictactoe import tictactoe, random_board, label_space
    
    learner = learner_class(num_outputs=5, label_names=label_space)
    
    correct = 0
    recent_correct = 0
    window = 50
    
    results = []
    
    for i in range(rounds):
        board = random_board()
        true_label = tictactoe(board)
        true_idx = label_space.index(true_label)
        
        pred_idx = learner.predict(board)
        
        is_correct = pred_idx == true_idx
        if is_correct:
            correct += 1
            recent_correct += 1
        
        learner.update_history(board, pred_idx, true_idx)
        
        # Track recent accuracy
        if i >= window:
            old_board, old_pred, old_true = learner.history[i - window]
            if old_pred == old_true:
                recent_correct -= 1
        
        if verbose and (i + 1) % 50 == 0:
            stats = learner.get_stats()
            recent_acc = recent_correct / min(window, i + 1)
            print(f"R{i+1:3d}: Total={correct/(i+1):.1%} Recent={recent_acc:.1%} | "
                  f"Rules={stats['rules']:4d} Struct={stats['structural']:2d} "
                  f"Elim={stats['eliminated']}")
        
        results.append({
            'round': i,
            'correct': is_correct,
            'total_acc': correct / (i + 1),
            'stats': learner.get_stats().copy()
        })
    
    return learner, results


def compare_variants():
    """Compare different label handling strategies."""
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from tictactoe import tictactoe, random_board, label_space
    
    print("="*60)
    print("Testing Enhanced Rule Learner on TicTacToe")
    print("="*60)
    
    learner, results = run_test(EnhancedRuleLearner, rounds=500)
    
    final_acc = sum(r['correct'] for r in results) / len(results)
    print(f"\nFinal overall accuracy: {final_acc:.1%}")
    print(f"Final stats: {learner.get_stats()}")
    
    print("\n--- Discovered Structural Rules ---")
    for name, rule in learner.get_structural_rules().items():
        print(f"  {name}: {rule}")
    
    print("\n--- Top 15 Regular Rules ---")
    for rule in learner.get_top_rules(15):
        label = learner.label_names[rule.output] if rule.output < len(learner.label_names) else str(rule.output)
        print(f"  [{rule.confidence:.2f}] {dict(rule.pattern)} -> {label} (supp={rule.support})")
    
    # Analyze by label
    print("\n--- Per-Label Accuracy (last 200 rounds) ---")
    label_correct = defaultdict(int)
    label_total = defaultdict(int)
    
    for r in results[-200:]:
        idx = r['round']
        obs, guess, true = learner.history[idx]
        label_total[true] += 1
        if guess == true:
            label_correct[true] += 1
    
    for label_idx in range(5):
        total = label_total[label_idx]
        correct = label_correct[label_idx]
        acc = correct / total if total > 0 else 0
        print(f"  {label_space[label_idx]:5s}: {acc:5.1%} ({correct}/{total})")


if __name__ == "__main__":
    compare_variants()
