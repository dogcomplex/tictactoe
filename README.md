# Pragmatic Semantic Reasoning System (MVP)

This project implements the Minimum Viable Product (MVP) of a modular, multi-strategy semantic reasoning system designed to identify underlying patterns or rules in input-output data.

The system is based on a meta-synthesis architecture where different reasoning strategies (initially, a SAT-based approach and a GPU-accelerated tensor-based approach) are orchestrated to generate and validate hypotheses about the data.

## Project Structure

*   **`main_mvp.py`**: The main script to run the reasoning system MVP on a task file.
*   **`framework/`**: Contains the core orchestration components:
    *   `data_structures.py`: Defines standard formats for tasks (`TaskSpec`) and results (`CandidateProgram`).
    *   `strategy_interface.py`: Defines the `Strategy` abstract base class.
    *   `feature_extractor.py`: Implements basic feature extraction from task data.
    *   `dispatcher.py`: Handles strategy registration and execution (`StrategyRegistry`, `SimpleDispatcher`).
    *   `verifier.py`: Verifies candidate hypotheses against task examples.
    *   `reporter.py`: Aggregates results and generates reports.
*   **`strategies/`**: Contains wrappers for specific reasoning strategies:
    *   `sat_wrapper.py`: Wraps the SAT-based hypothesis system (`SATHypothesesAlgorithm` from `few_shot_algs/`).
    *   `recipes_wrapper.py`: Wraps the tensor-based hypothesis system (`HypothesisManager` from `recipes.py`).
*   **`tasks/`**: Contains example task specification files in JSON format (e.g., `tictactoe_standard.json`).
*   **`tests/`**: Contains unit and integration tests (using `pytest`).
*   **`recipes.py`**: Original implementation of the tensor-based hypothesis system. (Dependency)
*   **`few_shot_algs/sat_hypotheses.py`**: Original implementation of the SAT-based hypothesis system. (Dependency)
*   **`tictactoe.py`**: Provides TicTacToe game logic and ground truth data. (Dependency)
*   **`requirements.txt`**: Lists Python package dependencies.
*   **`REQUIREMENTS.md`**: Detailed project requirements engineering document.

## Setup and Installation

These instructions assume you have Python 3.x installed.

**1. Create a Virtual Environment (Recommended)**

*   **Windows (Command Prompt/PowerShell):**
    ```bash
    # Navigate to the project root directory
    cd path\to\your\project

    # Create the virtual environment (named .venv)
    python -m venv .venv

    # Activate the virtual environment
    .\.venv\Scripts\activate
    ```
    *(To deactivate later, simply type `deactivate`)*

*   **macOS/Linux (Bash/Zsh):**
    ```bash
    # Navigate to the project root directory
    cd path/to/your/project

    # Create the virtual environment (named .venv)
    python3 -m venv .venv

    # Activate the virtual environment
    source .venv/bin/activate
    ```
    *(To deactivate later, simply type `deactivate`)*

**2. Install Dependencies**

Once your virtual environment is active, install the required packages:

```bash
pip install -r requirements.txt
```
*Note:* Installing `torch` might take some time. If you have a compatible NVIDIA GPU and want GPU acceleration for the `recipes` strategy, ensure you have the correct CUDA toolkit installed and follow the PyTorch installation instructions specific to your system ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) if the basic `pip install torch` doesn't provide GPU support.

*Note:* `python-sat` might require build tools (like a C++ compiler) on some systems. If installation fails, consult the `python-sat` documentation or install common build tools (e.g., `build-essential` on Debian/Ubuntu, Build Tools for Visual Studio on Windows).

## Running Tests

To ensure the framework components are working correctly, run the automated tests using `pytest`:

```bash
# Make sure you are in the project root directory and your venv is active
pytest
```

This command will discover and run all files starting with `test_` in the `tests/` directory.

## Running the MVP

To run the reasoning system on a specific task:

1.  Ensure your virtual environment is active.
2.  Use the `main_mvp.py` script, providing the path to a task JSON file.

**Example:**

```bash
python main_mvp.py tasks/tictactoe_standard.json
```

**Optional Arguments:**

*   `-o` or `--output_dir`: Specify a directory to save the results (defaults to `mvp_runs`).

```bash
python main_mvp.py tasks/tictactoe_standard.json -o my_custom_results
```

The script will:
1.  Load the task from the specified JSON file.
2.  Initialize the framework components and registered strategies (SAT and Recipes).
3.  Extract basic features from the task data.
4.  Run each strategy to generate candidate hypotheses (`CandidateProgram` objects).
5.  Verify each candidate against the task examples.
6.  Generate a JSON report and a console summary in a timestamped sub-directory within the specified output directory (e.g., `mvp_runs/run_tictactoe_standard_small_20231027_103000/`).

## Next Steps (Post-MVP)

Refer to the `REQUIREMENTS.md` file for the full project scope and future development roadmap, including adding more strategies, enhancing verification, implementing ranking, etc.
