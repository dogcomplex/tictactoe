# Few-Shot Game Rule Learner

A system for inferring game rules from observations using hypothesis elimination.

## Overview

This project implements a **few-shot learning system** that can:
1. Observe game states and their outcomes (win/lose/draw/ongoing)
2. Automatically discover the underlying rules (e.g., "three X's in a row = win")
3. Predict outcomes for new, unseen game states

### Key Results

| Game Variant | Accuracy | Line Patterns Discovered | Notes |
|-------------|----------|-------------------------|-------|
| Standard TicTacToe | 96.4% | 16/16 (all) | All 8 win conditions for each player |
| No-Diagonal TicTacToe | 97.0% | 12/12 (correct) | Correctly excludes diagonal wins |

The system **automatically adapts** to game variants without being told the rules.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AdaptiveLearner                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Line Pattern    │  │ Draw Pattern    │  │ Rule-Based  │ │
│  │ Discovery       │  │ Discovery       │  │ Learning    │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
│           │                    │                   │        │
│           ▼                    ▼                   ▼        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Priority-Based Prediction                  ││
│  │  1. Line patterns (wins)                                ││
│  │  2. Draw detection (full board, no wins)                ││
│  │  3. Default "ok" (has empty spaces)                     ││
│  │  4. Fallback to learned rules                           ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Usage

```python
from test_harness import setup_standard_games, TestHarness
from few_shot_algs.adaptive_learner import AdaptiveLearner

# Setup games
harness = setup_standard_games()

# Run test
result = harness.run_test(AdaptiveLearner, "tictactoe_standard", rounds=500)
harness.print_result(result)
```

## Algorithm Details

### Hypothesis Elimination Strategy

1. **Generate candidate rules** from observations
   - Pattern: `Dict[position -> required_state]` (e.g., `{0:'1', 1:'1', 2:'1'}` = top row all X)
   - Output: Label index (0=ok, 1=win1, 2=win2, 3=draw, 4=error)

2. **Eliminate rules** contradicted by new observations

3. **Predict** via priority-based voting among remaining valid rules

4. **Discover structural patterns** (line-based win conditions)

### Key Innovations

- **Adaptive rule generation**: More rules for rare labels (wins/draws)
- **Structural pattern discovery**: Automatically finds line-based win conditions
- **Priority-based prediction**: Structural rules checked before learned rules

## File Structure

```
locus/
├── tictactoe.py              # Game oracles and state generation
├── test_harness.py           # Evaluation framework
├── requirements.txt          # Dependencies
└── few_shot_algs/
    ├── few_shot_alg.py       # Base interface
    ├── rule_learner.py       # Basic version (85% acc)
    ├── enhanced_learner.py   # Improved (95% acc)
    └── adaptive_learner.py   # Production (96-97% acc)
```

## Roadmap

### Phase 1: Generalize to Arbitrary Board Games
- [ ] Support variable board sizes (4x4, 5x5, etc.)
- [ ] Discover arbitrary pattern types beyond lines
- [ ] Handle multi-player games (>2 players)
- [ ] Implement token-based production rules (LHS => RHS)

### Phase 2: Complex Rule Discovery
- [ ] Boolean combinations (AND/OR/NOT of patterns)
- [ ] Temporal rules (sequence of moves)
- [ ] Conditional rules (if X then Y else Z)
- [ ] Quantified patterns (exists/forall)

### Phase 3: SAT-Based Approach
- [ ] Encode rules as CNF clauses
- [ ] Use SAT solver for consistency checking
- [ ] Generate minimal rule sets via MaxSAT
- [ ] Handle contradictions gracefully

### Phase 4: Advanced Games
- [ ] Chess-like games (piece movement rules)
- [ ] Card games (hidden information)
- [ ] Stochastic games (dice, randomness)
- [ ] Resource management games

## Current Limitations

1. **Draw detection** is weak (0-33% accuracy) - needs more sophisticated pattern
2. **Hardcoded** for 3x3 boards (can extend to NxN)
3. **Line patterns only** - no complex boolean rules yet
4. **No production chain rules** (LHS => RHS token spending) implemented

## Dependencies

- Python 3.10+
- No external dependencies for core functionality
- Optional: `python-sat` for SAT-based approach (requires network)
