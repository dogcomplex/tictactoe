# Few-Shot Learning for Tic-Tac-Toe State Classification

This project implements and compares various few-shot learning algorithms for classifying Tic-Tac-Toe game states. The goal is to predict the game outcome or state (win, loss, draw, ongoing, or error) given a board configuration, with minimal training examples.

## Project Structure

- `few_shot.py`: Main script that sets up the problem, runs the algorithms, and evaluates their performance.
- `tictactoe.py`: Contains the Tic-Tac-Toe game logic and state classification functions.
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

Each algorithm is designed to learn from a small number of examples and predict the game state for new board configurations.

## How it works

1. The `ProblemSetupTicTacToe` class generates random Tic-Tac-Toe board states and their corresponding labels.
2. The `Tester` class runs multiple rounds of predictions for each algorithm, updating their knowledge base after each round.
3. Performance metrics (accuracy and computation time) are collected and visualized for each algorithm.

## Running the project

To run the project, execute the `few_shot.py` script:

```
python few_shot.py
```

This will run all implemented algorithms for a specified number of rounds and generate performance graphs.

## Output

The script generates three graphs:

1. `algorithm_accuracy.png`: Shows the cumulative accuracy of each algorithm over time.
2. `algorithm_compute_time.png`: Displays the average computation time for each algorithm per round.
3. `algorithm_efficiency.png`: Illustrates the efficiency (accuracy/time ratio) of each algorithm over rounds.

These graphs help visualize and compare the performance of different few-shot learning algorithms in the context of Tic-Tac-Toe state classification.

## Customization

You can modify the number of rounds, add new algorithms, or adjust the problem setup by editing the `few_shot.py` file.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- PyTorch
- scikit-learn
- transformers (for GPT-2)

Ensure all dependencies are installed before running the script.
