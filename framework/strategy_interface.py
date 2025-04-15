from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional # Added Optional
from .data_structures import TaskSpec, CandidateProgram

class Strategy(ABC):
    """
    Abstract Base Class for all reasoning strategies.
    """
    strategy_id: str = "base_strategy" # Unique identifier for the strategy

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy, potentially with configuration.
        """
        self.config = config or {}
        print(f"Initializing strategy: {self.strategy_id}")

    @abstractmethod
    def generate(self, task_spec: TaskSpec, features: Dict[str, Any]) -> List[CandidateProgram]:
        """
        Generate candidate programs/hypotheses based on the task specification
        and extracted features.

        Args:
            task_spec: The standardized task specification object.
            features: A dictionary of features extracted from the task data.

        Returns:
            A list of CandidateProgram objects representing the generated hypotheses.
        """
        pass

    def setup(self):
        """
        Optional method for any setup required before processing tasks
        (e.g., loading models, initializing resources).
        """
        print(f"Setting up strategy: {self.strategy_id}")
        pass

    def teardown(self):
        """
        Optional method for any cleanup required after processing
        (e.g., releasing resources).
        """
        print(f"Tearing down strategy: {self.strategy_id}")
        pass 