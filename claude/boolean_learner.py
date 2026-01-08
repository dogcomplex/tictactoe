"""
Boolean Hypothesis Elimination Learner

A principled approach based on SAT-style reasoning WITHOUT actually using
a SAT solver (since we don't have network access).

Key insight from SAT/production rules:
- Rules are implications: IF conditions THEN output
- A rule is "alive" until we see a counterexample
- Prediction = vote among alive rules that match

This learner:
1. Encodes each (position, symbol) pair as a boolean variable
2. Rules are conjunctions of literals => output label
3. Strictly eliminates rules on contradictions
4. No cheating - purely observational learning

Token Production Rule encoding:
- LHS tokens are "requirements" (must be present)
- RHS tokens are "productions" (what the rule produces)
- A token on both sides acts as a catalyst (IF condition)
"""

import random
from typing import List, Tuple, Dict, Set, Optional, Any, FrozenSet
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class BooleanCondition:
    """A condition: position P has symbol S (or doesn't)."""
    position: int
    symbol: str
    positive: bool = True  # If False, means "NOT this symbol"
    
    def matches(self, observation: str) -> bool:
        if self.position >= len(observation):
            return False
        has_symbol = observation[self.position] == self.symbol
        return has_symbol if self.positive else not has_symbol
    
    def __str__(self):
        neg = "" if self.positive else "!"
        return f"{neg}p{self.position}={self.symbol}"


@dataclass
class BooleanRule:
    """A rule: conjunction of conditions => output label."""
    conditions: FrozenSet[BooleanCondition]
    output: int
    alive: bool = True
    support: int = 0  # Times rule matched and was correct
    
    def matches(self, observation: str) -> bool:
        """Check if all conditions are satisfied."""
        return all(c.matches(observation) for c in self.conditions)
    
    def signature(self) -> Tuple:
        return (self.conditions, self.output)
    
    def __hash__(self):
        return hash(self.signature())
    
    def __str__(self):
        conds = " & ".join(str(c) for c in sorted(self.conditions, 
                                                   key=lambda c: (c.position, c.symbol)))
        return f"[{conds}] => {self.output}"


class BooleanHypothesisLearner:
    """
    Hypothesis elimination learner using boolean rules.
    
    Core algorithm:
    1. On each observation, generate new hypotheses consistent with it
    2. Kill any existing hypothesis contradicted by the observation  
    3. Predict via weighted vote of surviving hypotheses
    
    No cheating:
    - No knowledge of what symbols mean
    - No knowledge of board structure
    - No knowledge of what labels represent
    - Purely learns from (input, output) pairs
    """
    
    def __init__(self, num_outputs: int = 5, max_rules: int = 20000,
                 rules_per_obs: int = 200, max_conditions: int = 5,
                 use_negative_conditions: bool = True, **kwargs):
        self.num_outputs = num_outputs
        self.max_rules = max_rules
        self.rules_per_obs = rules_per_obs
        self.max_conditions = max_conditions
        self.use_negative_conditions = use_negative_conditions
        
        # Rule storage - indexed by signature for deduplication
        self.rules: Dict[Tuple, BooleanRule] = {}
        
        # Observations
        self.observations: List[Tuple[str, int]] = []
        self.history: List[Tuple[str, int, int]] = []
        
        # Discovered structure
        self.input_length: Optional[int] = None
        self.observed_symbols: Set[str] = set()
        self.label_counts: Dict[int, int] = defaultdict(int)
        self.label_observations: Dict[int, List[str]] = defaultdict(list)
        
        self.stats = {
            'generated': 0,
            'killed': 0,
            'predictions': 0,
        }
    
    def _generate_rules(self, observation: str, label: int, count: int):
        """Generate rules consistent with this observation."""
        if self.input_length is None:
            return
        
        generated = 0
        attempts = 0
        max_attempts = count * 3
        
        while generated < count and attempts < max_attempts:
            attempts += 1
            
            # Random number of conditions (bias toward more for rare labels)
            is_rare = self.label_counts[label] < 0.15 * sum(self.label_counts.values())
            min_conds = 2 if is_rare else 1
            max_conds = min(self.max_conditions + (2 if is_rare else 0), 
                           self.input_length)
            num_conds = random.randint(min_conds, max_conds)
            
            # Pick random positions
            positions = random.sample(range(self.input_length), num_conds)
            
            # Create conditions - mostly positive, some negative
            conditions = []
            for pos in positions:
                symbol = observation[pos]
                if self.use_negative_conditions and random.random() < 0.2:
                    # Negative condition: NOT some other symbol at this position
                    other_symbols = [s for s in self.observed_symbols if s != symbol]
                    if other_symbols:
                        neg_symbol = random.choice(other_symbols)
                        conditions.append(BooleanCondition(pos, neg_symbol, positive=False))
                    else:
                        conditions.append(BooleanCondition(pos, symbol, positive=True))
                else:
                    # Positive condition: this symbol at this position
                    conditions.append(BooleanCondition(pos, symbol, positive=True))
            
            # Create rule
            rule = BooleanRule(
                conditions=frozenset(conditions),
                output=label,
                support=1
            )
            
            sig = rule.signature()
            if sig not in self.rules:
                self.rules[sig] = rule
                generated += 1
                self.stats['generated'] += 1
    
    def _kill_contradicted(self, observation: str, label: int):
        """Kill rules contradicted by this observation."""
        killed = 0
        
        for sig, rule in list(self.rules.items()):
            if not rule.alive:
                continue
            
            if rule.matches(observation):
                if rule.output == label:
                    # Rule is confirmed
                    rule.support += 1
                else:
                    # Rule is contradicted - KILL IT
                    rule.alive = False
                    killed += 1
        
        self.stats['killed'] += killed
        
        # Remove dead rules periodically to save memory
        if killed > 100:
            self.rules = {sig: r for sig, r in self.rules.items() if r.alive}
    
    def _prune_rules(self):
        """Keep only the best alive rules."""
        alive_rules = [(sig, r) for sig, r in self.rules.items() if r.alive]
        
        if len(alive_rules) <= self.max_rules:
            return
        
        # Score by support and specificity
        scored = []
        for sig, rule in alive_rules:
            score = rule.support * (1 + len(rule.conditions) * 0.3)
            scored.append((score, sig))
        
        scored.sort(reverse=True)
        keep = set(sig for _, sig in scored[:self.max_rules])
        
        self.rules = {sig: r for sig, r in self.rules.items() 
                     if sig in keep or not r.alive}
    
    def predict(self, observation: str) -> int:
        """Predict by voting among alive matching rules."""
        self.stats['predictions'] += 1
        
        # Discover structure
        if self.input_length is None:
            self.input_length = len(observation)
        for char in observation:
            self.observed_symbols.add(char)
        
        # Collect votes from alive rules that match
        votes: Dict[int, float] = defaultdict(float)
        
        for rule in self.rules.values():
            if rule.alive and rule.matches(observation):
                # Weight by support and specificity
                weight = (1 + rule.support) * (1 + len(rule.conditions) * 0.5)
                votes[rule.output] += weight
        
        if votes:
            return max(votes, key=votes.get)
        
        # Fallback: most common label
        if self.label_counts:
            return max(self.label_counts, key=self.label_counts.get)
        
        return 0
    
    def update_history(self, observation: str, guess: int, correct_label: int):
        """Learn from observation."""
        self.history.append((observation, guess, correct_label))
        self.observations.append((observation, correct_label))
        self.label_counts[correct_label] += 1
        self.label_observations[correct_label].append(observation)
        
        # Discover structure
        if self.input_length is None:
            self.input_length = len(observation)
        for char in observation:
            self.observed_symbols.add(char)
        
        # Calculate how many rules to generate
        total = sum(self.label_counts.values())
        freq = self.label_counts[correct_label] / total if total > 0 else 1.0
        
        # Generate MANY more rules for rare labels
        if freq < 0.05:
            count = self.rules_per_obs * 10
        elif freq < 0.15:
            count = self.rules_per_obs * 5
        elif freq < 0.3:
            count = self.rules_per_obs * 2
        else:
            count = self.rules_per_obs
        
        # Generate new rules
        self._generate_rules(observation, correct_label, count)
        
        # Kill contradicted rules
        self._kill_contradicted(observation, correct_label)
        
        # Prune
        self._prune_rules()
    
    def get_stats(self) -> Dict[str, Any]:
        alive = sum(1 for r in self.rules.values() if r.alive)
        return {
            **self.stats,
            'rules': alive,
            'total_rules': len(self.rules),
            'observations': len(self.observations),
            'symbols': len(self.observed_symbols),
        }
    
    def describe_knowledge(self) -> str:
        lines = ["=== Boolean Hypothesis Learner ===\n"]
        
        alive = [r for r in self.rules.values() if r.alive]
        lines.append(f"Observations: {len(self.observations)}")
        lines.append(f"Alive rules: {len(alive)}")
        lines.append(f"Input length: {self.input_length}")
        lines.append(f"Symbols: {sorted(self.observed_symbols)}")
        
        lines.append("\n--- Label Distribution ---")
        total = sum(self.label_counts.values())
        for label in range(self.num_outputs):
            count = self.label_counts.get(label, 0)
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  Label {label}: {count} ({pct:.1f}%)")
        
        lines.append("\n--- Top Rules by Support ---")
        top = sorted(alive, key=lambda r: (-r.support, -len(r.conditions)))[:20]
        for rule in top:
            lines.append(f"  {rule} (support={rule.support})")
        
        # Group rules by output label
        lines.append("\n--- Rules per Label ---")
        by_label: Dict[int, List[BooleanRule]] = defaultdict(list)
        for rule in alive:
            by_label[rule.output].append(rule)
        
        for label in range(self.num_outputs):
            label_rules = by_label[label]
            lines.append(f"  Label {label}: {len(label_rules)} rules")
            # Show top 3 for each
            top_label = sorted(label_rules, key=lambda r: -r.support)[:3]
            for rule in top_label:
                lines.append(f"    {rule} (sup={rule.support})")
        
        return '\n'.join(lines)
    
    def get_top_rules_for_label(self, label: int, n: int = 10) -> List[BooleanRule]:
        """Get top rules predicting a specific label."""
        label_rules = [r for r in self.rules.values() 
                      if r.alive and r.output == label]
        return sorted(label_rules, key=lambda r: (-r.support, -len(r.conditions)))[:n]


def test_boolean_learner():
    """Test on standard TicTacToe."""
    import sys
    sys.path.insert(0, '/home/claude/locus')
    from tictactoe import tictactoe, random_board, label_space
    
    print("=" * 60)
    print("Boolean Hypothesis Learner Test")
    print("=" * 60)
    
    learner = BooleanHypothesisLearner(num_outputs=5, max_rules=20000)
    
    correct = 0
    per_label = defaultdict(lambda: [0, 0])
    
    for i in range(500):
        board = random_board()
        true_label = tictactoe(board)
        true_idx = label_space.index(true_label)
        
        pred_idx = learner.predict(board)
        
        is_correct = pred_idx == true_idx
        if is_correct:
            correct += 1
            per_label[true_label][0] += 1
        per_label[true_label][1] += 1
        
        learner.update_history(board, pred_idx, true_idx)
        
        if (i + 1) % 100 == 0:
            stats = learner.get_stats()
            print(f"R{i+1:3d}: {correct/(i+1):.1%} | "
                  f"alive={stats['rules']} killed={stats['killed']}")
    
    print(f"\nFinal: {correct/500:.1%}")
    
    print("\nPer-label accuracy:")
    for label in label_space:
        c, t = per_label[label]
        if t > 0:
            print(f"  {label}: {c}/{t} = {c/t:.1%}")
    
    print(learner.describe_knowledge())


if __name__ == "__main__":
    test_boolean_learner()
