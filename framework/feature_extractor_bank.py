import logging
from typing import Dict, Callable, Any
from .data_structures import TaskSpec

logger = logging.getLogger(__name__)

class FeatureExtractorBank:
    """
    Manages and runs a collection of feature extractor functions.
    """
    def __init__(self):
        """Initializes the bank with an empty registry of extractors."""
        self.extractors: Dict[str, Callable[[TaskSpec], Any]] = {}
        logger.info("Initialized FeatureExtractorBank.")

    def register(self, name: str, func: Callable[[TaskSpec], Any]):
        """
        Registers a feature extractor function.

        Args:
            name: The name to assign to the feature.
            func: The function to call (should accept a TaskSpec).
        """
        if name in self.extractors:
            raise ValueError(f"Feature extractor '{name}' already registered.")
        self.extractors[name] = func
        logger.info(f"Registered feature extractor: {name}")

    def run(self, task_spec: TaskSpec) -> Dict[str, Any]:
        """
        Runs all registered feature extractors on the given task spec.

        Args:
            task_spec: The TaskSpec to extract features from.

        Returns:
            A dictionary where keys are feature names and values are the
            extracted feature values. If an extractor fails, its value will be None.
        """
        features: Dict[str, Any] = {}
        logger.info(f"Running {len(self.extractors)} feature extractors...")
        for name, func in self.extractors.items():
            try:
                result = func(task_spec)
                features[name] = result
                logger.debug(f"  Extractor '{name}' result: {result}") # Use debug level
            except Exception as e:
                logger.error(f"Error running feature extractor '{name}': {e}", exc_info=True) # Match test expectation
                features[name] = None # Store None on error
        logger.info("Feature extraction complete.")
        return features 