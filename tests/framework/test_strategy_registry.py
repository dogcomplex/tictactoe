import pytest
# Add project root to sys.path to allow framework import
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from framework.strategy_registry import StrategyRegistry
from framework.strategy_interface import Strategy # Import base Strategy for typing/mocking
from framework.data_structures import TaskSpec, CandidateProgram

# --- Mock Strategy ---
class MockStrategy(Strategy):
    strategy_id = "MockStrategy_v1"

    def generate(self, task_spec: TaskSpec, features) -> list[CandidateProgram]:
        # Simple mock implementation
        return [CandidateProgram("mock_repr", "MockType", self.strategy_id)]

    def setup(self):
        pass

    def teardown(self):
        pass

# --- Tests ---

def test_strategy_registry_init():
    """Test basic initialization of the registry."""
    registry = StrategyRegistry()
    assert registry.get_all_strategies() == {}

def test_strategy_registry_register():
    """Test registering a single strategy."""
    registry = StrategyRegistry()
    mock_strategy = MockStrategy()
    registry.register(mock_strategy)
    assert registry.get_strategy("MockStrategy_v1") == mock_strategy
    assert len(registry.get_all_strategies()) == 1

def test_strategy_registry_register_duplicate():
    """Test that registering a strategy with the same ID raises an error."""
    registry = StrategyRegistry()
    mock_strategy1 = MockStrategy()
    mock_strategy2 = MockStrategy() # Same ID
    registry.register(mock_strategy1)
    with pytest.raises(ValueError, match="Strategy ID 'MockStrategy_v1' already registered."):
        registry.register(mock_strategy2)

def test_strategy_registry_get():
    """Test retrieving a registered strategy."""
    registry = StrategyRegistry()
    mock_strategy = MockStrategy()
    registry.register(mock_strategy)
    retrieved = registry.get_strategy("MockStrategy_v1")
    assert retrieved == mock_strategy

def test_strategy_registry_get_all():
    """Test retrieving all registered strategies."""
    registry = StrategyRegistry()
    mock_strategy = MockStrategy()
    # Register another mock strategy with a different ID if needed for a more complex test
    registry.register(mock_strategy)
    all_strategies = registry.get_all_strategies()
    assert len(all_strategies) == 1
    assert "MockStrategy_v1" in all_strategies
    assert all_strategies["MockStrategy_v1"] == mock_strategy

def test_strategy_registry_get_nonexistent():
    """Test retrieving a non-existent strategy raises KeyError."""
    registry = StrategyRegistry()
    with pytest.raises(KeyError, match="Strategy with ID 'NonExistentStrategy' not found."):
        registry.get_strategy("NonExistentStrategy") 