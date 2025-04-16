import pytest
from unittest.mock import MagicMock, call
import sys
import os
from typing import List, Dict, Any

# Add project root for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from framework.strategy_interface import Strategy
from framework.data_structures import TaskSpec, CandidateProgram

# --- Dummy Classes (If real ones not found) ---
class DummyDispatcher:
    def __init__(self, registry): # Use 'registry' to match real class
        self.registry = registry # Use 'registry'

    def dispatch(self, task_spec: TaskSpec, features: Dict[str, Any]) -> List[CandidateProgram]: # Match method name
        # Minimal dummy logic based on real class
        all_candidates = []
        # Assume registry has get_all_strategies() -> Dict[str, Strategy]
        strategies_dict = self.registry.get_all_strategies()
        for strategy_id, strategy in strategies_dict.items():
            try:
                candidates = strategy.generate(task_spec, features)
                all_candidates.extend(candidates)
            except Exception:
                pass # Ignore errors in dummy
        # Return flattened list, as per real class
        return all_candidates

class MockStrategy(Strategy): # Inherit from Strategy to be type-correct
    strategy_id: str
    generates: List[CandidateProgram]

    def __init__(self, strategy_id="mock", generates=None):
        super().__init__() # Call parent __init__ if needed
        self.strategy_id = strategy_id
        self.generates = generates if generates is not None else []
        self.setup_called = False
        self.teardown_called = False
        # Mock the generate method directly
        self.generate = MagicMock(return_value=self.generates)

    # Add the missing abstract method implementation
    def generate(self, task_spec: TaskSpec, features: Dict[str, Any]) -> List[CandidateProgram]:
        pass # Since generate is mocked in __init__, this body won't be hit

# --- Import Real or Use Dummy ---
try:
    from framework.dispatcher import Dispatcher
except ImportError:
    print("Warning: framework.dispatcher not found. Using dummy class.")
    Dispatcher = DummyDispatcher

# --- Fixtures ---
@pytest.fixture
def mock_registry():
    """Fixture for a mock StrategyRegistry."""
    registry = MagicMock()
    # Mock the method used by the actual Dispatcher
    registry.get_all_strategies.return_value = {} # Default to empty dict
    return registry

@pytest.fixture
def mock_aggregator(): # Keep this fixture if other tests need it
    """Fixture for a mock ResultAggregator."""
    aggregator = MagicMock()
    aggregator.aggregate_and_verify.side_effect = lambda task, lists: [c for sublist in lists for c in sublist]
    return aggregator

@pytest.fixture
def sample_task_spec():
    """Fixture for a sample TaskSpec."""
    return TaskSpec(task_id="dispatch_test", inputs=[{"data": 1}], outputs=[{"result": 2}])

@pytest.fixture
def sample_features():
    """Fixture for sample features."""
    return {"feature1": True}

# --- Tests ---

def test_dispatcher_init(mock_registry, mock_aggregator):
    """Test Dispatcher initialization."""
    dispatcher = Dispatcher(mock_registry) # Pass only registry
    assert dispatcher.registry == mock_registry # Check for 'registry' attribute

def test_dispatcher_dispatch_one_strategy(mock_registry, sample_task_spec, sample_features):
    """Test dispatching with a single strategy."""
    mock_candidate = CandidateProgram("repr1", "MockType", "MockStrategy_1")
    mock_strategy = MockStrategy(strategy_id="MockStrategy_1", generates=[mock_candidate])

    # Configure the mock registry to return the mock strategy
    # Dispatcher uses get_all_strategies() which returns a dict
    mock_registry.get_all_strategies.return_value = {"MockStrategy_1": mock_strategy}

    dispatcher = Dispatcher(mock_registry) # Pass only registry
    # Use the correct method name: dispatch
    results = dispatcher.dispatch(sample_task_spec, sample_features)

    # Assert strategy's generate was called
    mock_strategy.generate.assert_called_once_with(sample_task_spec, sample_features)

    # Check results (Dispatcher flattens the list)
    assert results == [mock_candidate]

def test_dispatcher_dispatch_multiple_strategies(mock_registry, sample_task_spec, sample_features):
    """Test dispatching with multiple strategies."""
    mock_candidate1 = CandidateProgram("repr1", "MockType", "MockStrategy_1")
    mock_candidate2 = CandidateProgram("repr2", "MockType", "MockStrategy_2")
    mock_strategy1 = MockStrategy(strategy_id="MockStrategy_1", generates=[mock_candidate1])
    mock_strategy2 = MockStrategy(strategy_id="MockStrategy_2", generates=[mock_candidate2])

    mock_registry.get_all_strategies.return_value = {
        "MockStrategy_1": mock_strategy1,
        "MockStrategy_2": mock_strategy2
    }

    dispatcher = Dispatcher(mock_registry) # Pass only registry
    results = dispatcher.dispatch(sample_task_spec, sample_features) # Use dispatch

    mock_strategy1.generate.assert_called_once_with(sample_task_spec, sample_features)
    mock_strategy2.generate.assert_called_once_with(sample_task_spec, sample_features)

    # Check order might not be guaranteed, check presence
    assert len(results) == 2
    assert mock_candidate1 in results
    assert mock_candidate2 in results
    # assert list_of_candidate_lists == [[mock_candidate1], [mock_candidate2]] # Old assertion

def test_dispatcher_dispatch_no_strategies(mock_registry, sample_task_spec, sample_features):
    """Test dispatching with no strategies available."""
    mock_registry.get_all_strategies.return_value = {} # Returns empty dict
    dispatcher = Dispatcher(mock_registry) # Pass only registry
    results = dispatcher.dispatch(sample_task_spec, sample_features) # Use dispatch
    assert results == []

def test_dispatcher_dispatch_strategy_error(mock_registry, sample_task_spec, sample_features, caplog):
    """Test dispatching when one strategy raises an error."""
    mock_candidate1 = CandidateProgram("repr1", "MockType", "MockStrategy_1")
    mock_strategy1 = MockStrategy(strategy_id="MockStrategy_1", generates=[mock_candidate1])
    mock_strategy2 = MockStrategy(strategy_id="ErrorStrategy", generates=[]) # Changed ID for clarity
    # Make strategy 2's generate raise an exception
    mock_strategy2.generate.side_effect = ValueError("Strategy 2 failed!")

    mock_registry.get_all_strategies.return_value = {
        "MockStrategy_1": mock_strategy1,
        "ErrorStrategy": mock_strategy2
    }

    dispatcher = Dispatcher(mock_registry) # Pass only registry
    results = dispatcher.dispatch(sample_task_spec, sample_features) # Use dispatch

    # Check that strategy 1 still generated its candidate list
    assert results == [mock_candidate1] # Only successful strategy returns results
    # Check that strategy 2's generate was still called
    mock_strategy2.generate.assert_called_once_with(sample_task_spec, sample_features)
    # Check logger output for the error
    assert "Strategy ErrorStrategy failed" in caplog.text # Check for correct strategy ID in log
    assert "Strategy 2 failed!" in caplog.text

# Remove test_dispatcher_uses_aggregator as Dispatcher no longer holds aggregator
# ... (rest of file unchanged) 