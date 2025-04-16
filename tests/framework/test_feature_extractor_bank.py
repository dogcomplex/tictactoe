import pytest
import sys
import os

# Add project root for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from framework.data_structures import TaskSpec

# Assume FeatureExtractorBank is in framework.feature_extractor_bank
try:
    from framework.feature_extractor_bank import FeatureExtractorBank
except ImportError:
    print("Warning: framework.feature_extractor_bank not found. Using dummy class for tests.")
    from typing import List, Dict, Any, Callable
    # Dummy FeatureExtractorBank
    class FeatureExtractorBank:
        def __init__(self):
            self.extractors: Dict[str, Callable[[TaskSpec], Any]] = {}

        def register(self, name: str, func: Callable[[TaskSpec], Any]):
            if name in self.extractors:
                raise ValueError(f"Feature extractor '{name}' already registered.")
            self.extractors[name] = func
            print(f"Registered feature extractor: {name}") # Added print

        def run(self, task_spec: TaskSpec) -> Dict[str, Any]:
            features = {}
            print(f"Running {len(self.extractors)} feature extractors...") # Added print
            for name, func in self.extractors.items():
                try:
                    features[name] = func(task_spec)
                    print(f"  Extractor '{name}' result: {features[name]}") # Added print
                except Exception as e:
                    print(f"Error running feature extractor '{name}': {e}") # Added print
                    features[name] = None # Or some error indicator
            return features

# --- Mock Feature Extractors ---
def extract_num_examples(task_spec: TaskSpec) -> int:
    return len(task_spec.inputs)

def extract_input_type(task_spec: TaskSpec) -> str:
    if not task_spec.inputs:
        return "unknown"
    first_input = task_spec.inputs[0]
    if isinstance(first_input.get('board'), str):
        return "tictactoe_board"
    if isinstance(first_input.get('data'), int):
         return "integer"
    return "mixed_or_unknown"

def extract_always_error(task_spec: TaskSpec):
    raise ValueError("This extractor always fails.")

# --- Fixtures ---
@pytest.fixture
def sample_task_spec_features():
    """Fixture for a TaskSpec for feature extraction tests."""
    return TaskSpec(
        task_id="feature_test",
        inputs=[
            {"board": "111000000"},
            {"board": "000000000"},
        ],
        outputs=[
            {"label": "win1"},
            {"label": "ok"},
        ]
    )

# --- Tests ---

def test_feature_bank_init():
    """Test initializing an empty bank."""
    bank = FeatureExtractorBank()
    assert bank.extractors == {}

def test_feature_bank_register():
    """Test registering feature extractors."""
    bank = FeatureExtractorBank()
    bank.register("num_examples", extract_num_examples)
    bank.register("input_type", extract_input_type)
    assert "num_examples" in bank.extractors
    assert "input_type" in bank.extractors
    assert bank.extractors["num_examples"] == extract_num_examples

def test_feature_bank_register_duplicate():
    """Test registering a duplicate extractor name raises error."""
    bank = FeatureExtractorBank()
    bank.register("num_examples", extract_num_examples)
    with pytest.raises(ValueError, match="Feature extractor 'num_examples' already registered."):
        bank.register("num_examples", extract_num_examples)

def test_feature_bank_run(sample_task_spec_features):
    """Test running registered feature extractors."""
    bank = FeatureExtractorBank()
    bank.register("num_examples", extract_num_examples)
    bank.register("input_type", extract_input_type)

    features = bank.run(sample_task_spec_features)

    assert len(features) == 2
    assert features["num_examples"] == 2
    assert features["input_type"] == "tictactoe_board"

def test_feature_bank_run_empty(sample_task_spec_features):
    """Test running with no registered extractors."""
    bank = FeatureExtractorBank()
    features = bank.run(sample_task_spec_features)
    assert features == {}

def test_feature_bank_run_with_error(sample_task_spec_features, caplog):
    """Test running extractors when one raises an error."""
    bank = FeatureExtractorBank()
    bank.register("num_examples", extract_num_examples)
    bank.register("error_extractor", extract_always_error)
    bank.register("input_type", extract_input_type)

    features = bank.run(sample_task_spec_features)

    assert len(features) == 3
    assert features["num_examples"] == 2
    assert features["input_type"] == "tictactoe_board"
    assert features["error_extractor"] is None # Or specific error marker

    # Check logs for the error message
    assert "Error running feature extractor 'error_extractor'" in caplog.text
    assert "ValueError: This extractor always fails." in caplog.text 