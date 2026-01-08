"""
Few-Shot Learning Test Harness

A unified framework for testing and comparing different learners
on various game variants.
"""

import random
import time
from typing import List, Dict, Callable, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import sys

sys.path.insert(0, '/home/claude/locus')


@dataclass
class GameConfig:
    """Configuration for a game oracle."""
    name: str
    oracle: Callable[[str], str]
    board_generator: Callable[[], str]
    label_space: List[str]
    board_size: int = 9
    description: str = ""


@dataclass 
class TestResult:
    """Result of a single test run."""
    game_name: str
    learner_name: str
    rounds: int
    final_accuracy: float
    accuracy_over_time: List[float]
    discoveries: Dict[str, Any]
    runtime_seconds: float
    per_label_accuracy: Dict[str, float]


class TestHarness:
    """Unified test harness for few-shot learners."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.games: Dict[str, GameConfig] = {}
        self.results: List[TestResult] = []
    
    def register_game(self, config: GameConfig):
        """Register a game configuration."""
        self.games[config.name] = config
        if self.verbose:
            print(f"Registered game: {config.name}")
    
    def run_test(self, learner_class, game_name: str, rounds: int = 500,
                 learner_kwargs: Dict = None) -> TestResult:
        """Run a test of a learner on a game."""
        if game_name not in self.games:
            raise ValueError(f"Unknown game: {game_name}")
        
        game = self.games[game_name]
        learner_kwargs = learner_kwargs or {}
        
        # Initialize learner
        learner = learner_class(
            num_outputs=len(game.label_space),
            label_names=game.label_space,
            board_size=game.board_size,
            **learner_kwargs
        )
        
        # Run evaluation
        start_time = time.time()
        correct = 0
        accuracy_over_time = []
        label_correct = defaultdict(int)
        label_total = defaultdict(int)
        
        for i in range(rounds):
            board = game.board_generator()
            true_label = game.oracle(board)
            true_idx = game.label_space.index(true_label)
            
            pred_idx = learner.predict(board)
            
            is_correct = pred_idx == true_idx
            if is_correct:
                correct += 1
                label_correct[true_label] += 1
            label_total[true_label] += 1
            
            learner.update_history(board, pred_idx, true_idx)
            
            if (i + 1) % 50 == 0:
                acc = correct / (i + 1)
                accuracy_over_time.append(acc)
                if self.verbose:
                    stats = learner.get_stats() if hasattr(learner, 'get_stats') else {}
                    lines = stats.get('lines', stats.get('lines_discovered', 0))
                    rules = stats.get('rules', 0)
                    print(f"  R{i+1:3d}: {acc:.1%} | Rules={rules} Lines={lines}")
        
        runtime = time.time() - start_time
        
        # Calculate per-label accuracy
        per_label_acc = {}
        for label in game.label_space:
            total = label_total[label]
            correct_count = label_correct[label]
            per_label_acc[label] = correct_count / total if total > 0 else 0.0
        
        # Get discoveries
        discoveries = {}
        if hasattr(learner, 'discovered_lines'):
            discoveries['line_patterns'] = list(learner.discovered_lines.keys())
        if hasattr(learner, 'draw_rule') and learner.draw_rule:
            discoveries['has_draw'] = True
        if hasattr(learner, 'describe_knowledge'):
            discoveries['knowledge'] = learner.describe_knowledge()
        
        result = TestResult(
            game_name=game_name,
            learner_name=learner_class.__name__,
            rounds=rounds,
            final_accuracy=correct / rounds,
            accuracy_over_time=accuracy_over_time,
            discoveries=discoveries,
            runtime_seconds=runtime,
            per_label_accuracy=per_label_acc
        )
        
        self.results.append(result)
        return result
    
    def print_result(self, result: TestResult):
        """Print a formatted test result."""
        print(f"\n{'='*60}")
        print(f"Result: {result.learner_name} on {result.game_name}")
        print(f"{'='*60}")
        print(f"Final Accuracy: {result.final_accuracy:.1%}")
        print(f"Runtime: {result.runtime_seconds:.2f}s")
        print(f"\nPer-Label Accuracy:")
        for label, acc in result.per_label_accuracy.items():
            print(f"  {label:8s}: {acc:.1%}")
        
        if 'line_patterns' in result.discoveries:
            patterns = result.discoveries['line_patterns']
            print(f"\nDiscovered {len(patterns)} line patterns:")
            for p in patterns[:10]:  # Show first 10
                print(f"  - {p}")
            if len(patterns) > 10:
                print(f"  ... and {len(patterns) - 10} more")
        
        if result.discoveries.get('has_draw'):
            print("\n✓ Discovered draw condition")
    
    def compare_learners(self, learner_classes: List, game_name: str,
                        rounds: int = 500) -> Dict[str, TestResult]:
        """Compare multiple learners on a game."""
        print(f"\n{'#'*60}")
        print(f"Comparing learners on: {game_name}")
        print(f"{'#'*60}")
        
        results = {}
        for learner_class in learner_classes:
            print(f"\n--- Testing {learner_class.__name__} ---")
            result = self.run_test(learner_class, game_name, rounds)
            results[learner_class.__name__] = result
        
        # Print comparison
        print(f"\n{'='*60}")
        print("Comparison Summary")
        print(f"{'='*60}")
        print(f"{'Learner':<25} {'Accuracy':>10} {'Runtime':>10}")
        print("-" * 45)
        for name, result in results.items():
            print(f"{name:<25} {result.final_accuracy:>9.1%} {result.runtime_seconds:>9.2f}s")
        
        return results


def setup_standard_games():
    """Set up standard TicTacToe game variants."""
    from tictactoe import tictactoe, tictactoe_no_diags, random_board, label_space, boards
    
    harness = TestHarness()
    
    # Standard TicTacToe
    harness.register_game(GameConfig(
        name="tictactoe_standard",
        oracle=tictactoe,
        board_generator=random_board,
        label_space=label_space,
        description="Standard TicTacToe with diagonal wins"
    ))
    
    # No-diagonal variant
    # Generate boards for no-diag variant
    nodiag_boards = {}
    def gen_nodiag(board='000000000'):
        if board in nodiag_boards:
            return
        label = tictactoe_no_diags(board)
        if label == 'error':
            return
        nodiag_boards[board] = label
        if label == 'ok':
            count = board.count('0')
            turn = '1' if count % 2 == 1 else '2'
            for i in range(9):
                if board[i] == '0':
                    new_board = board[:i] + turn + board[i+1:]
                    gen_nodiag(new_board)
    gen_nodiag()
    nodiag_board_list = list(nodiag_boards.keys())
    
    harness.register_game(GameConfig(
        name="tictactoe_no_diags",
        oracle=tictactoe_no_diags,
        board_generator=lambda: random.choice(nodiag_board_list),
        label_space=label_space,
        description="TicTacToe without diagonal wins"
    ))
    
    return harness


def main():
    """Run comprehensive tests."""
    from few_shot_algs.adaptive_learner import AdaptiveLearner
    from few_shot_algs.rule_learner import RuleLearner
    from few_shot_algs.enhanced_learner import EnhancedRuleLearner
    
    harness = setup_standard_games()
    
    # Test learners on both variants
    learners = [AdaptiveLearner]  # Could add more: RuleLearner, EnhancedRuleLearner
    
    for game_name in ["tictactoe_standard", "tictactoe_no_diags"]:
        for learner_class in learners:
            print(f"\n{'#'*60}")
            print(f"Testing {learner_class.__name__} on {game_name}")
            print(f"{'#'*60}")
            
            result = harness.run_test(learner_class, game_name, rounds=500)
            harness.print_result(result)


if __name__ == "__main__":
    main()
