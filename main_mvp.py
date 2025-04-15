import json
import time
import os
import argparse
import logging
import sys

# Explicitly add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# --- DIAGNOSTICS START ---
print("--- Python Path ---")
print(sys.path)
print("-------------------")
framework_dir = os.path.join(project_root, 'framework')
print(f"Checking framework dir: {framework_dir}")
print(f"Framework dir exists: {os.path.isdir(framework_dir)}")
if os.path.isdir(framework_dir):
    print(f"Framework dir contents: {os.listdir(framework_dir)}")
    reporter_file = os.path.join(framework_dir, 'reporter.py')
    print(f"Checking reporter file: {reporter_file}")
    print(f"Reporter file exists: {os.path.isfile(reporter_file)}")
try:
    import framework
    print("Successfully imported 'framework' package.")
except ImportError as e:
    print(f"Failed to import 'framework' package directly: {e}")
print("--- DIAGNOSTICS END ---")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure framework and strategies are importable
# (Assuming main_mvp.py is in the project root)
try:
    from framework.data_structures import TaskSpec
    from framework.feature_extractor import FeatureExtractorRunner
    from framework.dispatcher import StrategyRegistry, SimpleDispatcher
    from framework.verifier import Verifier
    from framework.reporter import ResultAggregator, Reporter
    from strategies.sat_wrapper import SATStrategyWrapper
    from strategies.recipes_wrapper import RecipesStrategyWrapper
    # Import tictactoe if needed for data generation (though reading from file is preferred)
    # from tictactoe import generate_all_answers, random_board, label_space
except ImportError as e:
     logger.error(f"Failed to import framework/strategy components: {e}. Ensure PYTHONPATH is set correctly or run from project root.", exc_info=True)
     sys.exit(1)

def main(task_file: str, output_dir: str):
    """
    Main orchestration logic for the MVP.
    """
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger.info(f"--- Starting MVP Run ({run_timestamp}) ---")
    logger.info(f"Task File: {task_file}")
    logger.info(f"Output Directory: {output_dir}")

    # --- 1. Load Task ---
    try:
        task_spec = TaskSpec.from_json_file(task_file)
        logger.info(f"Successfully loaded task: {task_spec.task_id}")
        logger.info(f"Description: {task_spec.description}")
        logger.info(f"Number of Examples: {len(task_spec.inputs)}")
    except Exception as e:
        logger.error(f"Failed to load task from {task_file}: {e}", exc_info=True)
        return # Cannot proceed without a task

    # --- 2. Initialize Components ---
    try:
        feature_runner = FeatureExtractorRunner()
        registry = StrategyRegistry()
        verifier = Verifier()
        # Initialize and register strategies
        # Pass configuration if needed, e.g., config={'use_disk_cache': True}
        sat_strategy = SATStrategyWrapper()
        recipes_strategy = RecipesStrategyWrapper(config={'use_disk_cache': True}) # Example config
        registry.register(sat_strategy)
        registry.register(recipes_strategy)

        # Setup strategies (e.g., load models, caches)
        registry.setup_all()

        dispatcher = SimpleDispatcher(registry)
        aggregator = ResultAggregator(verifier)
        reporter = Reporter(output_dir=os.path.join(output_dir, f"run_{task_spec.task_id}_{run_timestamp}"))
    except Exception as e:
        logger.error(f"Failed to initialize framework components: {e}", exc_info=True)
        return # Cannot proceed

    # --- 3. Run Workflow ---
    all_candidates = []
    try:
        # a. Extract Features
        features = feature_runner.run(task_spec)

        # b. Dispatch to Strategies
        all_candidates = dispatcher.run(task_spec, features)

        # c. Aggregate & Verify Results
        processed_candidates = aggregator.process_results(task_spec, all_candidates)

        # d. Generate Report
        reporter.generate_report(task_spec, processed_candidates, run_timestamp)

    except Exception as e:
        logger.error(f"An error occurred during the main workflow: {e}", exc_info=True)
    finally:
        # --- 4. Teardown ---
        logger.info("Tearing down strategies...")
        registry.teardown_all()
        logger.info(f"--- MVP Run Finished ({run_timestamp}) ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Semantic Reasoning System MVP.")
    parser.add_argument("task_file", help="Path to the JSON task specification file.")
    parser.add_argument("-o", "--output_dir", default="mvp_runs", help="Directory to save results (default: mvp_runs).")
    args = parser.parse_args()

    # Create output dir if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    main(args.task_file, args.output_dir) 