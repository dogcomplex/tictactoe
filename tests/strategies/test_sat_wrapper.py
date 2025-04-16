import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from strategies.sat_wrapper import SATStrategyWrapper
from framework.data_structures import TaskSpec, CandidateProgram, VerificationStatus

# Define a mock InternalHypothesis class matching the expected structure
class MockInternalHypothesis:
    def __init__(self, clauses, score=0.5, complexity=10, is_active=True):
        self.clauses = clauses # list[list[int]]
        self.score = score
        self.complexity = complexity
        self.posterior_prob = score # Use score as proxy for testing
        self.is_active = is_active

# --- Fixtures ---
@pytest.fixture
def sample_task_spec_sat():
    """Fixture for a TaskSpec suitable for SAT testing."""
    # Uses the standard 5 labels for TicTacToe
    return TaskSpec(
        task_id="sat_wrapper_test_ttt",
        inputs=[
            {"board": "111000000"},
            {"board": "000000000"},
            {"board": "121121212"}, # Draw example
        ],
        outputs=[
            {"label": "win1"}, # index 1
            {"label": "ok"},   # index 0
            {"label": "draw"}, # index 3
        ]
    )

@pytest.fixture
def sat_wrapper_instance():
    """Fixture to create a SATStrategyWrapper instance."""
    # Patch the internal algorithm during instantiation of the wrapper
    with patch('strategies.sat_wrapper.InternalSATHypothesesAlg') as MockInternalAlg:
        # Configure the mock instance if needed, e.g., mock_alg_instance = MockInternalAlg.return_value
        wrapper = SATStrategyWrapper()
        # Make the mock accessible for tests if needed
        wrapper._MockInternalAlgClass = MockInternalAlg
        wrapper._mock_internal_alg_instance = MockInternalAlg.return_value
        yield wrapper # Use yield to allow teardown check

# --- Tests ---

def test_sat_wrapper_init():
    """Test basic initialization."""
    wrapper = SATStrategyWrapper()
    assert wrapper.strategy_id == "SAT_Hypotheses_v1"
    assert wrapper.internal_algorithm is None
    assert wrapper.num_outputs == 5 # Default
    assert wrapper.label_space == []

def test_sat_wrapper_generate_instantiates_internal(sat_wrapper_instance, sample_task_spec_sat):
    """Test that generate instantiates the internal algorithm correctly."""
    wrapper = sat_wrapper_instance
    # Setup mock return value for active_hypotheses
    wrapper._mock_internal_alg_instance.active_hypotheses = []

    _ = wrapper.generate(sample_task_spec_sat, {})

    # Check that InternalSATHypothesesAlg was called (instantiated)
    wrapper._MockInternalAlgClass.assert_called_once()
    # Check instance is stored
    assert wrapper.internal_algorithm == wrapper._mock_internal_alg_instance
    # Check num_outputs was potentially passed or set (depends on internal alg structure)
    # For the patched version, we might check call args or attributes set on the mock instance
    # Example: wrapper._MockInternalAlgClass.assert_called_with(beam_width=5000)
    # Example: assert wrapper._mock_internal_alg_instance.num_outputs == 5
    assert wrapper.num_outputs == 5 # Check wrapper's num_outputs
    assert wrapper.label_space == ['ok', 'win1', 'win2', 'draw', 'error']

def test_sat_wrapper_generate_processes_examples(sat_wrapper_instance, sample_task_spec_sat):
    """Test that generate calls update_history on the internal algorithm."""
    wrapper = sat_wrapper_instance
    mock_internal_alg = wrapper._mock_internal_alg_instance
    mock_internal_alg.active_hypotheses = [] # Ensure it's iterable

    _ = wrapper.generate(sample_task_spec_sat, {})

    # Check update_history calls
    assert mock_internal_alg.update_history.call_count == 3
    # Check specific calls (board_state, guess_idx, correct_label_idx)
    mock_internal_alg.update_history.assert_any_call("111000000", 1, 1)
    mock_internal_alg.update_history.assert_any_call("000000000", 0, 0)
    mock_internal_alg.update_history.assert_any_call("121121212", 3, 3)

def test_sat_wrapper_generate_converts_hypotheses(sat_wrapper_instance, sample_task_spec_sat):
    """Test conversion of internal hypotheses to CandidateProgram objects."""
    wrapper = sat_wrapper_instance
    mock_internal_alg = wrapper._mock_internal_alg_instance

    # Create mock internal hypotheses
    hyp1_clauses = [[1, 2], [-3]]
    hyp2_clauses = [[-4, 5]]
    mock_hyp1 = MockInternalHypothesis(clauses=hyp1_clauses, score=0.8, complexity=3)
    mock_hyp2 = MockInternalHypothesis(clauses=hyp2_clauses, score=0.6, complexity=2)
    mock_internal_alg.active_hypotheses = [mock_hyp1, mock_hyp2]

    candidates = wrapper.generate(sample_task_spec_sat, {})

    assert len(candidates) == 2

    # Check candidate 1
    cand1 = candidates[0] if candidates[0].provenance['internal_score'] == 0.8 else candidates[1]
    assert cand1.program_representation == hyp1_clauses
    assert cand1.representation_type == 'CNF'
    assert cand1.source_strategy == wrapper.strategy_id
    assert cand1.confidence == 0.8
    assert cand1.verification_status == VerificationStatus.NOT_VERIFIED
    assert cand1.provenance['complexity'] == 3
    assert cand1.provenance['internal_score'] == 0.8

    # Check candidate 2
    cand2 = candidates[0] if candidates[0].provenance['internal_score'] == 0.6 else candidates[1]
    assert cand2.program_representation == hyp2_clauses
    assert cand2.representation_type == 'CNF'
    assert cand2.source_strategy == wrapper.strategy_id
    assert cand2.confidence == 0.6
    assert cand2.verification_status == VerificationStatus.NOT_VERIFIED
    assert cand2.provenance['complexity'] == 2
    assert cand2.provenance['internal_score'] == 0.6

def test_sat_wrapper_generate_handles_no_hypotheses(sat_wrapper_instance, sample_task_spec_sat):
    """Test generate when the internal algorithm returns no active hypotheses."""
    wrapper = sat_wrapper_instance
    mock_internal_alg = wrapper._mock_internal_alg_instance
    mock_internal_alg.active_hypotheses = []

    candidates = wrapper.generate(sample_task_spec_sat, {})

    assert candidates == []

def test_sat_wrapper_teardown(sat_wrapper_instance, sample_task_spec_sat):
    """Test that teardown cleans up the internal algorithm instance."""
    wrapper = sat_wrapper_instance
    # Run generate first to ensure internal_algorithm is instantiated
    wrapper._mock_internal_alg_instance.active_hypotheses = []
    _ = wrapper.generate(sample_task_spec_sat, {})
    assert wrapper.internal_algorithm is not None

    wrapper.teardown()
    assert wrapper.internal_algorithm is None

# Add test case for failing internal algorithm instantiation if needed
def test_sat_wrapper_generate_instantiation_failure(sample_task_spec_sat):
    """Test generate returns empty list if internal instantiation fails."""
    with patch('strategies.sat_wrapper.InternalSATHypothesesAlg') as MockInternalAlg:
        MockInternalAlg.side_effect = Exception("Failed to init!")
        wrapper = SATStrategyWrapper()
        candidates = wrapper.generate(sample_task_spec_sat, {})
        assert candidates == []
        assert wrapper.internal_algorithm is None 