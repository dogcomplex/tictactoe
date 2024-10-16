from abc import ABC, abstractmethod

# 2. Algorithm Interface
class Algorithm(ABC):
    def __init__(self):
        self.history = []  # Stores tuples of (observation, guess, correct_label)

    @abstractmethod
    def predict(self, observation: str) -> int:
        pass

    def update_history(self, observation: str, guess: int, correct_label: int):
        self.history.append((observation, guess, correct_label))
