import pytest
from unittest.mock import patch, MagicMock, call, ANY
import torch
import sys
import os
from pytest_mock import MockerFixture
import logging

# Add project root for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from framework.data_structures import TaskSpec, CandidateProgram, VerificationStatus
from strategies.recipes_wrapper import RecipesStrategyWrapper, map_standard_label_to_recipes_token
from recipes import HypothesisManager as InternalHypothesisManager
from recipes import map_observation_to_tensor as internal_map_observation

# --- Fixtures ---
@pytest.fixture
def mock_task_spec_recipes():
    """Fixture for a sample TaskSpec for recipes tests."""
    return TaskSpec(
        task_id="recipes_test",
        inputs=[
            {"board": ".X.O.X..O"}, # Example 1
            {"board": "X........"}, # Example 2
        ],
        outputs=[
            {"label": "ok"},      # Example 1 expected label
            {"label": "win1"},    # Example 2 expected label
        ]
    )

# Mapping for patching
MOCK_INTERNAL_INDEX_TO_TOKEN = {27: 'C', 28: 'W', 29: 'L', 30: 'D', 31: 'E'}
MOCK_LABEL_SPACE = ['ok', 'win1', 'win2', 'draw', 'error']

@pytest.fixture
def mock_internal_manager(mocker: MockerFixture):
    """Fixture for a mocked HypothesisManager."""
    mock_manager = mocker.MagicMock(spec=InternalHypothesisManager)
    mock_manager.device = torch.device('cpu')

    # Mock the valid_hypotheses attribute itself, not its sum
    mock_manager.valid_hypotheses = mocker.MagicMock() # Treat it as an object
    mock_manager.valid_hypotheses.__bool__.return_value = True # Allows checking `if manager.valid_hypotheses:`

    # Mock the get_top_hypotheses method to return the correct tuple format
    mock_tensor = mocker.MagicMock(spec=torch.Tensor)
    mock_index = 0
    mock_stats = {} # Or some relevant stats dictionary
    mock_manager.get_top_hypotheses.return_value = [(mock_tensor, mock_index, mock_stats)]

    # Mock the update_valid_hypotheses method (the correct method name)
    mock_manager.update_valid_hypotheses.return_value = None

    # Mock the filter_hypotheses method (if still used, otherwise remove)
    # mock_manager.filter_hypotheses.return_value = None # Comment out if not used

    # Mock the map_observation_to_tensor method (if it exists on the Manager, otherwise remove)
    # mock_map_result = mocker.MagicMock(spec=torch.Tensor) # Comment out if not used
    # mock_manager.map_observation_to_tensor.return_value = mock_map_result # Comment out if not used


    # Mock other necessary attributes/methods if needed based on wrapper usage
    # e.g., mock_manager.some_other_method.return_value = ...

    return mock_manager

@pytest.fixture
def recipes_wrapper_instance(mocker: MockerFixture, mock_internal_manager):
    """Fixture to create a RecipesStrategyWrapper instance with mocks."""
    # Patch the *actual* dependencies used *inside* recipes_wrapper.py
    # Use the aliased name 'InternalHypothesisManager' as used in the wrapper module
    with patch('strategies.recipes_wrapper.InternalHypothesisManager', return_value=mock_internal_manager) as mock_hm_init,\
         patch('strategies.recipes_wrapper.map_standard_label_to_recipes_token') as mock_mapper:
        # Instantiate the wrapper with a config
        wrapper = RecipesStrategyWrapper(config={'top_n_hypotheses': 2, 'use_disk_cache': False})
        # Call setup to initialize the internal manager (which uses the patched class)
        wrapper.setup() 

        # Assertions to check setup
        assert wrapper.internal_manager is not None, "Internal manager should be initialized after setup"
        assert wrapper.internal_manager == mock_internal_manager, "Mock manager not assigned correctly"
        # Check that the *patched* class was called correctly during setup
        mock_hm_init.assert_called_once_with(device=torch.device('cpu'), use_disk_cache=False)

        # Attach mocks for inspection in tests if needed
        wrapper._mock_map_label = mock_mapper
        wrapper._mock_logger = mocker.patch('strategies.recipes_wrapper.logger')
        wrapper._mock_hyp_manager_class = mock_hm_init
        wrapper._mock_internal_manager_instance = mock_internal_manager

        # Patch the internal tensor mapping function where it's imported
        mocker.patch('strategies.recipes_wrapper.internal_map_observation', return_value=torch.zeros(32, dtype=torch.bool))

        # Patch the label space imported from tictactoe
        mocker.patch('strategies.recipes_wrapper.ttt_label_space', MOCK_LABEL_SPACE, create=True)
        # Patch the internal token map used in _build_recipes_label_map
        mocker.patch('strategies.recipes_wrapper.internal_index_to_token', MOCK_INTERNAL_INDEX_TO_TOKEN, create=True)

        # Patch torch.sum *within the wrapper's scope* - default to 1 (valid)
        mocker.patch('strategies.recipes_wrapper.torch.sum').return_value.item.return_value = 1

        yield wrapper

# --- Tests ---

def test_recipes_wrapper_setup(recipes_wrapper_instance, mock_internal_manager):
    """Test if setup initializes the internal manager correctly."""
    # The fixture already performs setup and basic assertions
    assert recipes_wrapper_instance.internal_manager is not None
    assert recipes_wrapper_instance.internal_manager == mock_internal_manager
    # Verify InternalHypothesisManager was called correctly within the fixture's setup
    recipes_wrapper_instance._mock_hyp_manager_class.assert_called_once_with(
        device=torch.device('cpu'),
        use_disk_cache=False # From config passed in fixture
    )

def test_recipes_wrapper_generate_calls(recipes_wrapper_instance, mock_task_spec_recipes, mock_internal_manager, mocker: MockerFixture):
    """Test if generate processes examples and calls internal methods correctly."""
    
    # Configure mock manager to return 2 dummy hypotheses
    dummy_hypo1 = torch.zeros(32, dtype=torch.bool)
    dummy_hypo2 = torch.ones(32, dtype=torch.bool)
    mock_internal_manager.get_top_hypotheses.return_value = [dummy_hypo1, dummy_hypo2]
    # Configure visualize_hypothesis for these dummies
    mock_internal_manager.visualize_hypothesis.side_effect = lambda h: "DummyViz"

    # Re-patch internal_map_observation specifically for this test scope
    with patch('strategies.recipes_wrapper.internal_map_observation', return_value=torch.zeros(32, dtype=torch.bool)) as MockMapObs:
        # Configure the map_standard_label_to_recipes_token mock (via the instance)
        def map_side_effect(label, space):
            if label == 'ok': return 'C'
            if label == 'win1': return 'W'
            return None
        recipes_wrapper_instance._mock_map_label.side_effect = map_side_effect

        # Patch torch.sum within this test's scope to return 1 (valid)
        mocker.patch('strategies.recipes_wrapper.torch.sum').return_value.item.return_value = 1

        candidates = recipes_wrapper_instance.generate(mock_task_spec_recipes, {})

        # internal_map_observation should be called for each valid example
        expected_map_calls = [
            call(".X.O.X..O", "C"), # Example 1 input + mapped label ('ok' -> 'C')
            call("X........", "W"), # Example 2 input + mapped label ('win1' -> 'W')
        ]
        # Assert against the mock patched within this test's scope
        MockMapObs.assert_has_calls(expected_map_calls, any_order=False)

        # 2. Assert update_valid_hypotheses calls (the correct method)
        # It should be called with the tensor from map_observation and the *original label*
        expected_update_calls = [
            call(ANY, 'ok'), # Use ANY for tensor, expect original 'ok' label
            call(ANY, 'win1')  # Use ANY for tensor, expect original 'win1' label
        ]
        mock_internal_manager.update_valid_hypotheses.assert_has_calls(expected_update_calls, any_order=False)

        # 3. Assert get_top_hypotheses call
        mock_internal_manager.get_top_hypotheses.assert_called_once()
        call_args, call_kwargs = mock_internal_manager.get_top_hypotheses.call_args
        assert call_kwargs.get('n') == recipes_wrapper_instance.top_n_hypotheses
        assert call_kwargs.get('include_invalid') is False
        assert call_kwargs.get('include_miss') is True # Check the specific args we care about

        # 4. Assert candidate creation (basic check)
        # Now expect 2 candidates because get_top_hypotheses is mocked to return 2
        assert len(candidates) == 2 
        assert all(isinstance(c, CandidateProgram) for c in candidates)
        assert candidates[0].source_strategy == RecipesStrategyWrapper.strategy_id

def test_recipes_wrapper_generate_no_valid_hypotheses(recipes_wrapper_instance, mock_task_spec_recipes, mock_internal_manager):
    """Test generate returns empty list if no valid hypotheses remain after processing."""
    # Configure mock manager's get_top_hypotheses to return empty list
    mock_internal_manager.get_top_hypotheses.return_value = []

    candidates = recipes_wrapper_instance.generate(mock_task_spec_recipes, {})
    assert candidates == []
    # Should still process examples and call update_valid_hypotheses (the correct method)
    assert mock_internal_manager.update_valid_hypotheses.call_count == 2
    # Should still call get_top_hypotheses
    mock_internal_manager.get_top_hypotheses.assert_called_once()


def test_recipes_wrapper_generate_mapping_error(recipes_wrapper_instance, mock_task_spec_recipes, mock_internal_manager):
    """Test generate skips examples with unmappable labels and logs a warning."""
    mock_task_spec_recipes.outputs[0]['label'] = 'unknown_label' # Make first label unmappable

    # Configure the map_standard_label_to_recipes_token mock (via the instance)
    def map_side_effect(label, space):
        if label == 'unknown_label': return None
        if label == 'win1': return 'W'
        return 'C' # Default for others if any
    recipes_wrapper_instance._mock_map_label.side_effect = map_side_effect

    with patch('strategies.recipes_wrapper.internal_map_observation', return_value=torch.zeros(32, dtype=torch.bool)) as MockMapObs, \
         patch('strategies.recipes_wrapper.logger') as MockLoggerTestScope:

        candidates = recipes_wrapper_instance.generate(mock_task_spec_recipes, {})

        # Only the second example ('win1' -> 'W') should trigger internal_map_observation
        MockMapObs.assert_called_once_with("X........", "W")

        # Check that update_valid_hypotheses was only called for the second example
        # It should be called with the original label 'win1'
        mock_internal_manager.update_valid_hypotheses.assert_called_once_with(ANY, 'win1')

        # Check for warning log about skipping the first example
        MockLoggerTestScope.warning.assert_any_call(
            "Skipping example 0: Could not map label 'unknown_label' to recipes token."
        )


def test_recipes_wrapper_setup_failure(recipes_wrapper_instance):
    """Test generate returns empty list and logs error if setup failed (internal_manager is None)."""
    # Simulate setup failure by manually setting internal manager to None *after* fixture setup
    recipes_wrapper_instance.internal_manager = None
    # MockLoggerFixture = recipes_wrapper_instance._mock_logger # Logger from fixture

    # Re-patch logger within the test scope
    with patch('strategies.recipes_wrapper.logger') as MockLoggerTestScope:
        # Use a minimal dummy task spec
        dummy_task_spec = TaskSpec(task_id="dummy", inputs=[{}], outputs=[{}])
        candidates = recipes_wrapper_instance.generate(dummy_task_spec, {})

        assert candidates == []
        MockLoggerTestScope.error.assert_called_once_with(
            f"{recipes_wrapper_instance.strategy_id} setup failed or was not called. Cannot generate."
        )

def test_map_standard_label_to_recipes_token():
    """Test the label mapping helper function independently."""
    default_space = ['ok', 'win1', 'win2', 'draw', 'error']
    assert map_standard_label_to_recipes_token('ok', default_space) == 'C'
    assert map_standard_label_to_recipes_token('win1', default_space) == 'W'
    assert map_standard_label_to_recipes_token('win2', default_space) == 'W' # Assuming 'win1'/'win2' map to 'W'
    assert map_standard_label_to_recipes_token('draw', default_space) == 'D'
    assert map_standard_label_to_recipes_token('error', default_space) == 'E'
    assert map_standard_label_to_recipes_token('unknown', default_space) is None
    # Test with a non-default label space
    assert map_standard_label_to_recipes_token('ok', ['custom', 'labels']) is None

def test_recipes_wrapper_teardown(recipes_wrapper_instance, mock_internal_manager):
    """Test if teardown logs completion and potentially calls internal teardown."""
    # MockLoggerFixture = recipes_wrapper_instance._mock_logger # Logger from fixture
    # Add a teardown method to the mock manager if the wrapper calls it
    mock_internal_manager.teardown = MagicMock(name="MockManagerTeardown")

    # Re-patch logger within the test scope
    with patch('strategies.recipes_wrapper.logger') as MockLoggerTestScope:
        recipes_wrapper_instance.teardown()

        MockLoggerTestScope.info.assert_called_with(f"{recipes_wrapper_instance.strategy_id} torn down.") # Updated expected message
        # Assert if internal manager's teardown was called (if it exists and wrapper calls it)
        if hasattr(mock_internal_manager, 'teardown'):
            mock_internal_manager.teardown.assert_called_once()

def test_generate_candidates_valid_hypotheses(recipes_wrapper_instance, mock_task_spec_recipes, mock_internal_manager, mocker):
    """Test generating candidates when the internal manager returns valid hypotheses."""
    instance = recipes_wrapper_instance

    # Arrange: Mock internal manager methods
    hypo1_tensor = torch.rand(32) > 0.5 # Example tensor 1
    hypo2_tensor = torch.rand(32) > 0.5 # Example tensor 2
    mock_internal_manager.get_top_hypotheses.return_value = [
        # Simulate the tuple structure returned by the actual manager if needed
        # Assuming it returns (tensor, index, stats) or just tensor based on prev logs
         hypo1_tensor, hypo2_tensor
    ]
    mock_internal_manager.visualize_hypothesis.side_effect = lambda h: f"Viz:{h.sum().item()}"

    # Act
    candidates = instance.generate(mock_task_spec_recipes, {})

    # Assert
    assert len(candidates) == 2, "Should generate two candidates based on mock return"
    assert isinstance(candidates[0], CandidateProgram)
    assert isinstance(candidates[1], CandidateProgram)

    # Check source
    assert candidates[0].source_strategy == instance.strategy_id
    assert candidates[1].source_strategy == instance.strategy_id
    
    # Check representation type and content
    # The wrapper seems to store the raw tensor list as program_representation
    # and type as 'TensorRule' based on logs/implementation trace
    assert candidates[0].representation_type == 'TensorRule' 
    assert isinstance(candidates[0].program_representation, list) # Should be list form of tensor
    assert candidates[0].program_representation == hypo1_tensor.tolist()
    assert candidates[1].representation_type == 'TensorRule'
    assert isinstance(candidates[1].program_representation, list)
    assert candidates[1].program_representation == hypo2_tensor.tolist()

    # Check provenance contains the readable form
    assert 'readable_form' in candidates[0].provenance
    assert candidates[0].provenance['readable_form'] == f"Viz:{hypo1_tensor.sum().item()}"
    assert 'readable_form' in candidates[1].provenance
    assert candidates[1].provenance['readable_form'] == f"Viz:{hypo2_tensor.sum().item()}"

    # Check internal calls
    mock_internal_manager.get_top_hypotheses.assert_called_once()
    assert mock_internal_manager.visualize_hypothesis.call_count == 2
    first_call_arg = mock_internal_manager.visualize_hypothesis.call_args_list[0].args[0]
    second_call_arg = mock_internal_manager.visualize_hypothesis.call_args_list[1].args[0]
    assert isinstance(first_call_arg, torch.Tensor)
    assert isinstance(second_call_arg, torch.Tensor)
    # We cannot easily assert torch.equal(first_call_arg, hypo1_tensor) here due to mock comparison limits

# Potential further tests:
# - Test with 'use_disk_cache=True' in config (requires mocking file ops or manager differently)
# - Test error handling during hypothesis conversion loop (e.g., if get_top_hypotheses returns malformed data)
# - Test behavior if RECIPES_OUTPUT_TOKENS or recipes_num_tokens are missing/mismatched (if checks exist)
# - Test compatibility check logic in setup if it's more complex. 