from typing import Dict, Any, List, Tuple
import numpy as np
from .data_structures import TaskSpec

def get_input_types(task_spec: TaskSpec) -> Dict[str, str]:
    """
    Extracts the Python type of each unique input field name
    based on the first input example.
    """
    features = {}
    if not task_spec.inputs:
        return features

    first_input = task_spec.inputs[0]
    for key, value in first_input.items():
        features[f"input_type_{key}"] = type(value).__name__
    return features

def _get_dims(item: Any) -> List[int]:
    """Helper to recursively find dimensions of lists/arrays."""
    if isinstance(item, (list, np.ndarray)):
        if len(item) == 0:
            return [0]
        # Assume homogeneous lists/arrays for dim calculation
        sub_dims = _get_dims(item[0])
        return [len(item)] + sub_dims
    else:
        return []

def get_input_dimensions(task_spec: TaskSpec) -> Dict[str, List[int]]:
    """
    Extracts the dimensions of list or numpy array input fields
    based on the first input example.
    """
    features = {}
    if not task_spec.inputs:
        return features

    first_input = task_spec.inputs[0]
    for key, value in first_input.items():
        if isinstance(value, (list, np.ndarray)):
            dims = _get_dims(value)
            if dims: # Only add if it's actually list-like
               features[f"input_dims_{key}"] = dims
    return features

def get_value_ranges(task_spec: TaskSpec) -> Dict[str, Tuple[Any, Any]]:
    """
    Calculates the min and max values for numeric input fields across all examples.
    Handles strings by finding lexicographical min/max.
    Ignores non-numeric/non-string types or fields with mixed types.
    """
    ranges: Dict[str, Tuple[Any, Any]] = {}
    value_lists: Dict[str, List[Any]] = {}
    field_types: Dict[str, type] = {}

    if not task_spec.inputs:
        return {}

    # Collect all values and track types
    for input_example in task_spec.inputs:
        for key, value in input_example.items():
            current_type = type(value)
            if key not in value_lists:
                value_lists[key] = [value]
                field_types[key] = current_type
            else:
                value_lists[key].append(value)
                # If type changes, mark as mixed (None)
                if field_types[key] is not None and field_types[key] != current_type:
                    field_types[key] = None

    # Calculate ranges for non-mixed numeric or string fields
    for key, values in value_lists.items():
        field_type = field_types[key]
        if field_type is None: # Mixed types
            continue

        is_numeric = issubclass(field_type, (int, float))
        is_string = issubclass(field_type, str)

        if is_numeric:
            try:
                min_val = min(values)
                max_val = max(values)
                ranges[f"input_range_{key}"] = (min_val, max_val)
            except (ValueError, TypeError): # Handle potential errors if somehow non-numeric slip through
                 pass
        elif is_string:
             try:
                min_val = min(values)
                max_val = max(values)
                ranges[f"input_range_{key}"] = (min_val, max_val)
             except (ValueError, TypeError):
                 pass

    return ranges


# Feature Extractor Runner
class FeatureExtractorRunner:
    def __init__(self):
        # Register feature extraction functions here
        self.extractors = [
            get_input_types,
            get_input_dimensions,
            get_value_ranges,
            # Add more feature extractors here in the future
        ]
        print(f"Initialized FeatureExtractorRunner with {len(self.extractors)} extractors.")

    def run(self, task_spec: TaskSpec) -> Dict[str, Any]:
        """
        Runs all registered feature extractors on the task specification.

        Args:
            task_spec: The TaskSpec object.

        Returns:
            A dictionary containing all extracted features.
        """
        all_features: Dict[str, Any] = {}
        print(f"Running feature extraction for task: {task_spec.task_id}")
        for extractor in self.extractors:
            try:
                features = extractor(task_spec)
                all_features.update(features)
            except Exception as e:
                print(f"Warning: Feature extractor {extractor.__name__} failed: {e}")
        print(f"Feature extraction complete. Found {len(all_features)} features.")
        return all_features 