# Few-Shot Learning for Tic-Tac-Toe State Classification

This project implements and compares various few-shot learning algorithms for classifying Tic-Tac-Toe game states. The goal is to predict the game outcome or state (win, loss, draw, ongoing, or error) given a board configuration, with minimal training examples.

## Project Structure

- `few_shot.py`: Main script that sets up the problem, runs the algorithms, and evaluates their performance.
- `tictactoe.py`: Contains the Tic-Tac-Toe game logic and state classification functions.
- `recipes.py`: Contains the TicTacToeAlgorithm implementation.
- `few_shot_algs/`: Directory containing individual algorithm implementations:
  - `few_shot_alg.py`: Base class for all algorithms.
  - `random_forest.py`: Random Forest classifier.
  - `knn.py`: K-Nearest Neighbors classifier.
  - `prototypical_network.py`: Prototypical Network implementation.
  - `bayesian_nn.py`: Bayesian Neural Network.
  - `linear_regression.py`: Linear Regression classifier.
  - `siamese_network.py`: Siamese Network implementation.
  - `gaussian_process.py`: Gaussian Process classifier.
  - `transformer.py`: Transformer-based classifier.
  - `gpt2.py`: GPT-2 based few-shot learner.
  - `dqn.py`: Deep Q-Network implementation.
  - `forwardforward.py`: Forward-Forward Algorithm implementation.
  - `diffusion.py`: Diffusion Model implementation.
  - `distribution_approximator.py`: Distribution Approximator Algorithm.
  - `multi_armed_bandit.py`: Multi-Armed Bandit Algorithm.
  - `locus.py`: Locus Algorithm implementation.
  - `locus_bandit.py`: Locus Bandit Algorithm implementation.

## Algorithms

The project implements and compares the following few-shot learning algorithms:

1. Random Forest
2. K-Nearest Neighbors (KNN)
3. Prototypical Networks
4. Bayesian Neural Networks
5. Linear Regression
6. Siamese Networks
7. Gaussian Processes
8. Transformer
9. GPT-2
10. Deep Q-Network (DQN)
11. Forward-Forward Algorithm
12. Diffusion Model
13. Distribution Approximator
14. Multi-Armed Bandit
15. Locus Algorithm
16. Locus Bandit Algorithm

Each algorithm is designed to learn from a small number of examples and predict the game state for new board configurations.

## How it works

1. The `ProblemSetupRandom` class generates random Tic-Tac-Toe board states and their corresponding labels.
2. Each algorithm is initialized and trained on a small set of examples.
3. The algorithms are then tested on new, unseen board configurations.
4. Performance metrics (accuracy and computation time) are collected and compared for each algorithm.

## Running the project

To run the project, execute the `few_shot.py` script:

```
python few_shot.py
```

This will run all implemented algorithms for a specified number of rounds and generate performance comparisons.

## Output

The script generates performance metrics for each algorithm, including:

1. Accuracy: How well each algorithm predicts the correct game state.
2. Computation time: The time taken by each algorithm to make predictions.
3. Learning curve: How the accuracy of each algorithm improves with more examples.

## Customization

You can modify the number of rounds, add new algorithms, or adjust the problem setup by editing the `few_shot.py` file.

## Dependencies

- Python 3.x
- NumPy
- PyTorch
- scikit-learn
- transformers (for GPT-2)
- matplotlib (for visualization)

Ensure all dependencies are installed before running the script.

## Note

This project is designed for educational and research purposes to compare different few-shot learning approaches in the context of Tic-Tac-Toe state classification. The performance of these algorithms may vary depending on the specific problem setup and hyperparameters used. These are not the best possible implementations of these algorithms, but rather simple and clear versions to showcase the basic idea.  They still need tuning and optimization to be used in practice.
