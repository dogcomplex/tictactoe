import pytest
import json
import os
# Add project root to sys.path to allow framework import
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from framework.data_structures import TaskSpec # Assuming framework is in PYTHONPATH or adjacent

# Sample valid task data
valid_task_data = {
  "task_id": "test_task_valid",
  "description": "A valid test task.",
  "inputs": [ {"board": "111000000"} ],
  "outputs": [ {"label": "win1"} ],
  "constraints": [],
  "metadata": {}
}

# Sample invalid task data (inputs/outputs mismatch)
invalid_task_data = {
  "task_id": "test_task_invalid",
  "inputs": [ {"board": "111000000"} ],
  "outputs": [], # Mismatch
}

def test_taskspec_valid_creation():
    """Test creating TaskSpec from valid dictionary."""
    task = TaskSpec(**valid_task_data)
    assert task.task_id == "test_task_valid"
    assert len(task.inputs) == 1
    assert len(task.outputs) == 1

def test_taskspec_invalid_creation():
    """Test that creating TaskSpec with mismatched inputs/outputs raises ValueError."""
    with pytest.raises(ValueError, match="Number of inputs and outputs must match"):
        TaskSpec(**invalid_task_data)

def test_taskspec_empty_id():
    """Test that creating TaskSpec with empty task_id raises ValueError."""
    data = valid_task_data.copy()
    data["task_id"] = ""
    with pytest.raises(ValueError, match="task_id cannot be empty"):
        TaskSpec(**data)

def test_taskspec_from_json(tmp_path):
    """Test loading TaskSpec from a JSON file."""
    file_path = tmp_path / "task.json"
    with open(file_path, 'w') as f:
        json.dump(valid_task_data, f)

    task = TaskSpec.from_json_file(str(file_path))
    assert task.task_id == "test_task_valid"
    assert task.inputs[0]['board'] == "111000000"

def test_taskspec_to_json():
    """Test converting TaskSpec to JSON string."""
    task = TaskSpec(**valid_task_data)
    json_str = task.to_json()
    data = json.loads(json_str)
    assert data['task_id'] == "test_task_valid"
    assert data['inputs'] == valid_task_data['inputs'] 