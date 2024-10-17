from few_shot_algs.few_shot_alg import Algorithm
from recipes import TicTacToeAlgorithm

class LocusAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.ttt_algorithm = TicTacToeAlgorithm()

    def predict(self, observation: str) -> int:
        guess = self.ttt_algorithm.predict(observation)
        return ['C', 'W', 'L', 'D', 'E'].index(guess)

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        correct_label_str = ['C', 'W', 'L', 'D', 'E'][correct_label]
        self.ttt_algorithm.update_history(observation, ['C', 'W', 'L', 'D', 'E'][guess], correct_label_str)

