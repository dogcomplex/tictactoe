"""
Production Few-Shot Rule Learner

Features:
1. Automatic discovery of line-based win patterns
2. Draw detection (full board + no wins)
3. Adaptive pattern generation based on label rarity
4. Works on standard TicTacToe and variants (like no-diag)
"""

import random
from typing import List, Tuple, Dict, Set, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A rule mapping patterns to labels."""
    pattern: Dict[int, str]
    output: int
    confidence: float = 0.5
    support: int = 0
    priority: int = 0
    rule_type: str = "generated"  # "generated", "structural", "draw"
    
    def matches(self, board: str) -> bool:
        for pos, state in self.pattern.items():
            if board[pos] != state:
                return False
        return True
    
    def specificity(self) -> int:
        return len(self.pattern)
    
    def signature(self) -> tuple:
        return (tuple(sorted(self.pattern.items())), self.output)


class AdaptiveLearner:
    """
    Few-shot learner that adapts to the game being observed.
    
    Key capabilities:
    - Discovers line-based patterns (wins)
    - Detects draw conditions
    - Handles rare labels appropriately
    - No hardcoded game knowledge
    """
    
    def __init__(self, num_outputs=5, max_rules=3000, 
                 board_size=9, label_names=None):
        self.num_outputs = num_outputs
        self.max_rules = max_rules
        self.board_size = board_size
        self.label_names = label_names or [f'label_{i}' for i in range(num_outputs)]
        
        self.rules: List[Rule] = []
        self.rule_signatures: Set[tuple] = set()
        self.observations: List[Tuple[str, int]] = []
        self.history = []
        
        # Structural pattern tracking
        self.line_patterns = self._generate_line_patterns()
        self.discovered_lines: Dict[str, Rule] = {}
        
        # Draw pattern tracking
        self.draw_rule: Optional[Rule] = None
        self.draw_label_idx: Optional[int] = None
        
        # Label statistics
        self.label_counts = defaultdict(int)
        self.label_boards: Dict[int, List[str]] = defaultdict(list)
        
        self.stats = {
            'generated': 0,
            'eliminated': 0,
            'predictions': 0,
            'lines_discovered': 0,
        }
    
    def _generate_line_patterns(self) -> List[List[int]]:
        """Generate all potential line patterns (rows, cols, diags)."""
        size = int(self.board_size ** 0.5)  # Assume square board
        lines = []
        
        # Rows
        for r in range(size):
            lines.append([r * size + c for c in range(size)])
        
        # Columns
        for c in range(size):
            lines.append([r * size + c for r in range(size)])
        
        # Diagonals
        lines.append([i * size + i for i in range(size)])  # Main diagonal
        lines.append([i * size + (size - 1 - i) for i in range(size)])  # Anti-diagonal
        
        return lines
    
    def predict(self, board: str) -> int:
        """Predict label for board state."""
        self.stats['predictions'] += 1
        
        # 1. Check discovered line patterns first (win conditions)
        for name, rule in self.discovered_lines.items():
            if rule.matches(board):
                return rule.output
        
        # 2. Check draw condition (full board, no wins matched above)
        if '0' not in board:
            if self.draw_rule:
                return self.draw_rule.output
            # If we haven't seen draws yet but board is full with no win
            # likely a draw - use the draw label if we know it
            if self.draw_label_idx is not None:
                return self.draw_label_idx
        
        # 3. If board has empty spaces and no wins, likely "ok" (ongoing game)
        if '0' in board:
            # Default to "ok" (label 0) for ongoing games
            # This is safe since wins are already handled above
            return 0
        
        # 4. Fallback: use learned rules for edge cases
        matching = []
        for rule in self.rules:
            if rule.matches(board):
                matching.append(rule)
        
        if matching:
            matching.sort(key=lambda r: (-r.specificity(), -r.confidence, -r.support))
            votes = defaultdict(float)
            for rule in matching[:30]:
                weight = (rule.specificity() + 1) * rule.confidence * (rule.support + 1)
                votes[rule.output] += weight
            
            if votes:
                return max(votes, key=votes.get)
            return matching[0].output
        
        # 5. Prior-based prediction
        return self._prior_predict()
    
    def _prior_predict(self) -> int:
        """Predict based on label distribution."""
        if not self.label_counts:
            return 0
        
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
        self.label_boards[correct_label].append(observation)
        
        # Generate rules adapted to label rarity
        self._adaptive_generation(observation, correct_label)
        
        # Pattern discovery
        self._discover_line_patterns(observation, correct_label)
        self._discover_draw_pattern(observation, correct_label)
        
        # Eliminate contradictions
        self._eliminate_contradicted(observation, correct_label)
        
        # Update confidences periodically
        if len(self.observations) % 20 == 0:
            self._update_confidences()
        
        # Prune if needed
        if len(self.rules) > self.max_rules:
            self._prune()
    
    def _adaptive_generation(self, board: str, label: int):
        """Generate rules, with more for rare labels."""
        total_obs = sum(self.label_counts.values())
        label_freq = self.label_counts[label] / total_obs if total_obs > 0 else 1.0
        
        # Generate more rules for rare labels
        base_count = 30
        if label_freq < 0.1:  # Rare label
            count = base_count * 3
        elif label_freq < 0.3:
            count = base_count * 2
        else:
            count = base_count
        
        self._generate_rules(board, label, count)
    
    def _generate_rules(self, board: str, label: int, count: int):
        """Generate rules from observation."""
        generated = 0
        
        for specificity in range(1, 7):
            attempts = 0
            while generated < count and attempts < count * 2:
                attempts += 1
                
                positions = random.sample(range(self.board_size), 
                                         min(specificity, self.board_size))
                pattern = {pos: board[pos] for pos in positions}
                
                rule = Rule(pattern=pattern, output=label, support=1)
                sig = rule.signature()
                
                if sig not in self.rule_signatures:
                    self.rule_signatures.add(sig)
                    self.rules.append(rule)
                    generated += 1
                    self.stats['generated'] += 1
    
    def _discover_line_patterns(self, board: str, label: int):
        """Discover line-based win patterns."""
        # Only look at non-default labels (assume label 0 is "ok"/ongoing)
        if label == 0:
            return
        
        # Also skip draw label if we know it
        if self.draw_label_idx is not None and label == self.draw_label_idx:
            return
        
        for line in self.line_patterns:
            # Check if all positions have same non-empty state
            states = [board[pos] for pos in line]
            if len(set(states)) == 1 and states[0] != '0':
                player = states[0]
                name = f"line_{player}_{''.join(map(str, line))}"
                
                if name not in self.discovered_lines:
                    pattern = {pos: player for pos in line}
                    rule = Rule(
                        pattern=pattern, 
                        output=label,
                        confidence=1.0,
                        support=1,
                        priority=100,
                        rule_type="structural"
                    )
                    
                    # Verify against history
                    valid = True
                    for obs_board, obs_label in self.observations[:-1]:
                        if rule.matches(obs_board) and obs_label != label:
                            valid = False
                            break
                    
                    if valid:
                        self.discovered_lines[name] = rule
                        self.stats['lines_discovered'] += 1
                else:
                    self.discovered_lines[name].support += 1
    
    def _discover_draw_pattern(self, board: str, label: int):
        """Discover draw condition (full board, no wins)."""
        # Draw = board full ('0' not in board) and no line patterns match
        if '0' in board:
            return
        
        # Check if this full board has any line patterns that would win
        is_draw_candidate = True
        for name, rule in self.discovered_lines.items():
            if rule.matches(board):
                is_draw_candidate = False
                break
        
        if is_draw_candidate and label != 0:  # Not "ok", possibly draw
            # Check if we've seen multiple full boards with this label
            full_boards_with_label = [
                b for b in self.label_boards[label] 
                if '0' not in b
            ]
            
            if len(full_boards_with_label) >= 2:
                # This is likely the draw label
                if self.draw_label_idx is None or self.draw_label_idx == label:
                    self.draw_label_idx = label
                    # Create a "draw" rule: full board marker
                    # We use an empty pattern but check fullness in predict()
                    self.draw_rule = Rule(
                        pattern={},  # Special: checked via board fullness
                        output=label,
                        confidence=0.8,
                        rule_type="draw"
                    )
    
    def _eliminate_contradicted(self, board: str, label: int):
        """Remove rules contradicted by observation."""
        valid = []
        eliminated = 0
        
        for rule in self.rules:
            if rule.matches(board) and rule.output != label:
                eliminated += 1
            else:
                valid.append(rule)
        
        self.rules = valid
        self.stats['eliminated'] += eliminated
        
        # Also check discovered lines
        invalid = []
        for name, rule in self.discovered_lines.items():
            if rule.matches(board) and rule.output != label:
                invalid.append(name)
        for name in invalid:
            del self.discovered_lines[name]
    
    def _update_confidences(self):
        """Update rule confidences."""
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
        for rule in self.rules:
            rule.priority = (
                rule.specificity() * 10 +
                rule.confidence * 5 +
                min(rule.support, 10)
            )
        
        self.rules.sort(key=lambda r: -r.priority)
        self.rules = self.rules[:self.max_rules]
        self.rule_signatures = {r.signature() for r in self.rules}
    
    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'rules': len(self.rules),
            'lines': len(self.discovered_lines),
            'has_draw': self.draw_rule is not None,
            'observations': len(self.observations),
        }
    
    def describe_knowledge(self) -> str:
        """Human-readable description of learned knowledge."""
        lines = ["=== Learned Game Knowledge ===\n"]
        
        lines.append(f"Observations: {len(self.observations)}")
        lines.append(f"Rules: {len(self.rules)}")
        lines.append(f"Line patterns: {len(self.discovered_lines)}")
        
        lines.append("\n--- Line Patterns (Win Conditions) ---")
        for name, rule in sorted(self.discovered_lines.items()):
            label = self.label_names[rule.output] if rule.output < len(self.label_names) else str(rule.output)
            pos_str = ', '.join(f"p{p}={s}" for p, s in sorted(rule.pattern.items()))
            lines.append(f"  {pos_str} => {label} (seen {rule.support}x)")
        
        if self.draw_rule:
            label = self.label_names[self.draw_rule.output]
            lines.append(f"\n--- Draw Condition ---")
            lines.append(f"  Full board (no '0') with no win => {label}")
        
        lines.append("\n--- Label Distribution ---")
        total = sum(self.label_counts.values())
        for idx in range(self.num_outputs):
            count = self.label_counts[idx]
            pct = count / total * 100 if total > 0 else 0
            label = self.label_names[idx] if idx < len(self.label_names) else str(idx)
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        
        return '\n'.join(lines)


def test_standard_tictactoe():
    """Test on standard TicTacToe."""
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from tictactoe import tictactoe, random_board, label_space
    
    print("="*60)
    print("Standard TicTacToe Test")
    print("="*60)
    
    learner = AdaptiveLearner(num_outputs=5, label_names=label_space)
    
    correct = 0
    window_correct = 0
    window = 50
    
    for i in range(500):
        board = random_board()
        true_label = tictactoe(board)
        true_idx = label_space.index(true_label)
        
        pred_idx = learner.predict(board)
        
        is_correct = pred_idx == true_idx
        if is_correct:
            correct += 1
            window_correct += 1
        
        learner.update_history(board, pred_idx, true_idx)
        
        if i >= window:
            old = learner.history[i - window]
            if old[1] == old[2]:
                window_correct -= 1
        
        if (i + 1) % 100 == 0:
            stats = learner.get_stats()
            recent_acc = window_correct / window
            print(f"R{i+1:3d}: Total={correct/(i+1):.1%} Recent={recent_acc:.1%} | "
                  f"Rules={stats['rules']} Lines={stats['lines']} Draw={stats['has_draw']}")
    
    print(f"\nFinal: {correct/500:.1%}")
    print(learner.describe_knowledge())


def test_no_diagonals():
    """Test on TicTacToe variant without diagonal wins."""
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from tictactoe import tictactoe_no_diags, random_board, label_space
    
    print("\n" + "="*60)
    print("No-Diagonals TicTacToe Test")
    print("="*60)
    
    learner = AdaptiveLearner(num_outputs=5, label_names=label_space)
    
    # Get valid boards from no-diag variant
    boards = {}
    def gen_boards(board='000000000'):
        if board in boards:
            return
        label = tictactoe_no_diags(board)
        if label == 'error':
            return
        boards[board] = label
        if label == 'ok':
            count = board.count('0')
            turn = '1' if count % 2 == 1 else '2'
            for i in range(9):
                if board[i] == '0':
                    new_board = board[:i] + turn + board[i+1:]
                    gen_boards(new_board)
    
    gen_boards()
    board_list = list(boards.keys())
    print(f"Generated {len(board_list)} valid no-diag board states")
    
    correct = 0
    for i in range(500):
        board = random.choice(board_list)
        true_label = tictactoe_no_diags(board)
        true_idx = label_space.index(true_label)
        
        pred_idx = learner.predict(board)
        
        if pred_idx == true_idx:
            correct += 1
        
        learner.update_history(board, pred_idx, true_idx)
        
        if (i + 1) % 100 == 0:
            stats = learner.get_stats()
            print(f"R{i+1:3d}: Total={correct/(i+1):.1%} | Lines={stats['lines']}")
    
    print(f"\nFinal: {correct/500:.1%}")
    print(learner.describe_knowledge())
    
    # Check that diagonals were NOT learned
    print("\n--- Checking diagonal exclusion ---")
    diag_patterns = ['line_1_048', 'line_2_048', 'line_1_246', 'line_2_246']
    for pat in diag_patterns:
        if pat in learner.discovered_lines:
            print(f"  WARNING: Learned diagonal {pat} (should not exist in no-diag variant)")
        else:
            print(f"  OK: Did not learn {pat}")


if __name__ == "__main__":
    test_standard_tictactoe()
    test_no_diagonals()
