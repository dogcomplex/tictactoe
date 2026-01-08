"""
Variant Testing for Few-Shot Learners

Test learners on various game rule variants to detect cheating.
A learner that "knows" TicTacToe will fail on novel variants.
"""

import random
import sys
sys.path.insert(0, '/home/claude/locus')

from tictactoe import label_space
from few_shot_algs.blind_learner import BlindLearner, BlindLearnerV2
from few_shot_algs.adaptive_learner import AdaptiveLearner


def make_variant_oracle(win_conditions):
    """Create an oracle with custom win conditions."""
    def oracle(board):
        # Check for wins
        for condition in win_conditions:
            if all(board[i] == '1' for i in condition):
                return 'win1'
        
        for condition in win_conditions:
            if all(board[i] == '2' for i in condition):
                return 'win2'
        
        # Check for draw (full board, no wins)
        if '0' not in board:
            return 'draw'
        
        # Check for valid turn count
        count_1 = board.count('1')
        count_2 = board.count('2')
        if count_1 < count_2 or count_1 > count_2 + 1:
            return 'error'
        
        return 'ok'
    
    return oracle


# Standard TicTacToe win conditions
STANDARD_WINS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
    [0, 4, 8], [2, 4, 6],              # Diagonals
]

# No-diagonal variant
NO_DIAG_WINS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
]

# Only diagonal variant (weird!)
ONLY_DIAG_WINS = [
    [0, 4, 8], [2, 4, 6],
]

# Corners + center win
CORNER_CENTER_WINS = [
    [0, 4, 8], [2, 4, 6],  # Diagonals
    [0, 2, 4], [4, 6, 8],  # Corner-center combinations
]

# L-shaped wins
L_SHAPE_WINS = [
    [0, 1, 3], [1, 2, 5], [3, 6, 7], [5, 7, 8],  # L shapes
    [0, 2, 1], [2, 0, 3], [6, 8, 7], [8, 6, 3],  # More L shapes
]


def edge_oracle(board):
    """Win if you fill any edge of the board."""
    edges = [
        [0, 1, 2],  # Top
        [6, 7, 8],  # Bottom
        [0, 3, 6],  # Left
        [2, 5, 8],  # Right
    ]
    
    for edge in edges:
        if all(board[i] == '1' for i in edge):
            return 'win1'
        if all(board[i] == '2' for i in edge):
            return 'win2'
    
    count_1 = board.count('1')
    count_2 = board.count('2')
    if count_1 < count_2 or count_1 > count_2 + 1:
        return 'error'
    
    if '0' not in board:
        return 'draw'
    
    return 'ok'


def count_oracle(board):
    """Win if you have exactly 4 pieces on the board."""
    count_1 = board.count('1')
    count_2 = board.count('2')
    
    if count_1 < count_2 or count_1 > count_2 + 1:
        return 'error'
    
    if count_1 == 4:
        return 'win1'
    if count_2 == 4:
        return 'win2'
    
    if '0' not in board:
        return 'draw'
    
    return 'ok'


def generate_boards(oracle, max_boards=5000):
    """Generate all valid boards for a game variant."""
    boards = {}
    
    def explore(board='000000000'):
        if board in boards:
            return
        
        label = oracle(board)
        if label == 'error':
            return
        
        boards[board] = label
        
        if label == 'ok':
            count = board.count('0')
            turn = '1' if (9 - count) % 2 == 0 else '2'
            for i in range(9):
                if board[i] == '0':
                    new_board = board[:i] + turn + board[i+1:]
                    if len(boards) < max_boards:
                        explore(new_board)
    
    explore()
    return boards


def test_variant(learner_class, oracle, variant_name, rounds=500, verbose=True):
    """Test a learner on a specific variant."""
    boards = generate_boards(oracle)
    board_list = list(boards.keys())
    
    if verbose:
        print(f"\n{variant_name}: {len(board_list)} valid boards")
    
    learner = learner_class(num_outputs=5, board_size=9, label_names=label_space)
    
    correct = 0
    
    for i in range(rounds):
        board = random.choice(board_list)
        true_label = oracle(board)
        true_idx = label_space.index(true_label)
        
        pred_idx = learner.predict(board)
        
        if pred_idx == true_idx:
            correct += 1
        
        learner.update_history(board, pred_idx, true_idx)
    
    return correct / rounds


def run_comprehensive_test():
    """Run tests on all variants with all learners."""
    print("="*70)
    print("COMPREHENSIVE VARIANT TESTING")
    print("="*70)
    
    variants = [
        ("Standard TicTacToe", make_variant_oracle(STANDARD_WINS)),
        ("No-Diagonal", make_variant_oracle(NO_DIAG_WINS)),
        ("Only-Diagonal", make_variant_oracle(ONLY_DIAG_WINS)),
        ("Corner-Center", make_variant_oracle(CORNER_CENTER_WINS)),
        ("L-Shapes", make_variant_oracle(L_SHAPE_WINS)),
        ("Edge Fill", edge_oracle),
        ("Count-4 Win", count_oracle),
    ]
    
    learners = [
        ("Adaptive", AdaptiveLearner),
        ("Blind", BlindLearner),
        ("BlindV2", BlindLearnerV2),
    ]
    
    results = {}
    
    for variant_name, oracle in variants:
        results[variant_name] = {}
        for learner_name, learner_class in learners:
            acc = test_variant(learner_class, oracle, variant_name, rounds=300, verbose=False)
            results[variant_name][learner_name] = acc
    
    # Summary table
    print(f"\n{'Variant':<20}", end="")
    for learner_name, _ in learners:
        print(f"{learner_name:>12}", end="")
    print()
    print("-" * 60)
    
    for variant_name, _ in variants:
        print(f"{variant_name:<20}", end="")
        for learner_name, _ in learners:
            acc = results[variant_name][learner_name]
            print(f"{acc:>11.1%}", end="")
        print()
    
    # Detect cheating
    print("\n" + "="*70)
    print("CHEATING ANALYSIS")
    print("="*70)
    
    adaptive_std = results["Standard TicTacToe"]["Adaptive"]
    adaptive_novel = results["Count-4 Win"]["Adaptive"]
    blind_std = results["Standard TicTacToe"]["Blind"]
    blind_novel = results["Count-4 Win"]["Blind"]
    
    print(f"\nStandard TicTacToe - Adaptive: {adaptive_std:.1%}, Blind: {blind_std:.1%}")
    print(f"Count-4 Win (novel) - Adaptive: {adaptive_novel:.1%}, Blind: {blind_novel:.1%}")
    
    drop_adaptive = adaptive_std - adaptive_novel
    drop_blind = blind_std - blind_novel
    
    print(f"\nAccuracy drop on novel variant:")
    print(f"  Adaptive: {drop_adaptive:.1%}")
    print(f"  Blind: {drop_blind:.1%}")
    
    if drop_adaptive > drop_blind + 0.1:
        print("\n⚠️  AdaptiveLearner shows MORE degradation on novel rules!")
        print("   This suggests embedded TicTacToe knowledge (cheating).")
    elif drop_adaptive < drop_blind - 0.1:
        print("\n✓ AdaptiveLearner generalizes BETTER than blind learner.")
        print("   Its structure helps it discover novel patterns too.")
    else:
        print("\n~ Both learners degrade similarly on novel rules.")


if __name__ == "__main__":
    run_comprehensive_test()
