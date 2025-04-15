from typing import List, Dict, Any, Optional, Tuple
import logging
import sys
import os

# Add the parent directory (project root) to the Python path
# This allows importing framework and existing algorithm modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from framework.strategy_interface import Strategy
from framework.data_structures import TaskSpec, CandidateProgram, VerificationStatus

# Attempt to import the existing algorithm and its components
try:
    from few_shot_algs.sat_hypotheses import SATHypothesesAlgorithm as InternalSATHypothesesAlg
    from few_shot_algs.sat_hypotheses import Hypothesis as InternalHypothesis
    # We need pysat if the internal alg uses it, ensure it's installed
    from pysat.formula import CNF # Used for type hinting if needed
except ImportError as e:
    print(f"Error importing SATHypothesesAlgorithm or pysat: {e}. Ensure the algorithm exists and pysat is installed ('pip install python-sat'). Using placeholder.")
    # Define functional placeholder classes if import fails

    # Placeholder for the actual SAT algorithm implementation
    # Used when the real few_shot_algs.sat_hypotheses cannot be imported.
    class InternalSATHypothesesAlg:
        def __init__(self, num_outputs=None, beam_width=5000):
            # Initialize any necessary attributes
            self.num_outputs = num_outputs # Example attribute
            self.hypotheses = [] # Placeholder for storing hypotheses
            # Add active_hypotheses attribute consistent with usage in generate
            self.active_hypotheses = []

        def update_history(self, board_state, guess, correct_label): # Match signature used in generate
            """
            Placeholder method to update the algorithm's history.
            In a real implementation, this would process the example and update the SAT clauses.
            """
            logger.debug(f"InternalAlg Placeholder: Received update: board={board_state}, guess={guess}, correct={correct_label}")
            # Simple placeholder logic: maybe just store the example?
            # Or perhaps try to crudely update some internal state if needed for testing
            # Example: Add a dummy hypothesis based on the first example
            if not self.hypotheses:
                dummy_hyp = InternalHypothesis() # Use the dummy InternalHypothesis below
                dummy_hyp.clauses = [[1, 2], [-3]] # Example CNF
                dummy_hyp.score = 0.5
                # Ensure clauses are stored as list[list[int]] as expected by wrapper
                if isinstance(dummy_hyp.clauses, CNF):
                    dummy_hyp.clauses = dummy_hyp.clauses.clauses
                self.hypotheses.append(dummy_hyp)
                self.active_hypotheses.append(dummy_hyp) # Keep active_hypotheses consistent
            pass # Minimal placeholder action

        # Add placeholder for get_hypothesis_stats if generate method uses it
        def get_hypothesis_stats(self):
            logger.debug("InternalAlg Placeholder: get_hypothesis_stats called.")
            return {
                'total_active': len(self.active_hypotheses),
                'total_rejected': 0, # Placeholder
                'active_hypotheses': self.active_hypotheses
            }

    class InternalHypothesis:
        def __init__(self):
            self.clauses = [] # Placeholder for CNF clauses (list[list[int]])
            self.score = 0.0 # Placeholder score
            self.complexity = 0 # Placeholder complexity
            self.posterior_prob = 0.0 # Placeholder probability
            self.is_active = True # Placeholder status

    class CNF:
        # Minimal dummy CNF class if needed by placeholder logic
        def __init__(self, from_clauses=None):
            self.clauses = from_clauses if from_clauses is not None else []

logger = logging.getLogger(__name__)

# TODO: Refactor or make configurable parts of the internal BinaryConverter logic
# For now, define helper functions based on its logic, assuming TicTacToe structure
INPUT_VAR_COUNT = 9 # 9 squares
INPUT_BITS_PER_VAR = 3 # 0=Empty(100), 1=X(010), 2=O(001)
# Output count needs to be determined from TaskSpec or label_space
DEFAULT_OUTPUT_COUNT = 5 # From ttt label_space ['ok', 'win1', 'win2', 'draw', 'error']

def ttt_input_to_binary(input_str: str) -> str:
    """Converts 9-char TTT string to 27-bit binary string."""
    if len(input_str) != INPUT_VAR_COUNT:
        raise ValueError(f"Input string must be {INPUT_VAR_COUNT} characters long.")
    mapping = {'0': '100', '1': '010', '2': '001'}
    return ''.join(mapping.get(char, '000') for char in input_str) # Use '000' for unexpected chars

def ttt_output_to_binary(output_idx: int, num_outputs: int) -> str:
    """Converts label index to one-hot binary string."""
    if not 0 <= output_idx < num_outputs:
         raise ValueError(f"Output index {output_idx} out of range for {num_outputs} outputs.")
    return ''.join('1' if i == output_idx else '0' for i in range(num_outputs))

class SATStrategyWrapper(Strategy):
    """
    Wrapper for the existing SATHypothesesAlgorithm to conform to the
    standard Strategy interface.
    """
    strategy_id: str = "SAT_Hypotheses_v1"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.internal_algorithm: Optional[InternalSATHypothesesAlg] = None
        self.num_outputs = DEFAULT_OUTPUT_COUNT # Default, can be updated in generate
        self.label_space: List[str] = [] # To store the order for index mapping
        # Configurable parameters for the internal algorithm can be passed via self.config
        # e.g., self.config.get('max_clause_size', 2)

    def setup(self):
        """Initialize the internal SAT algorithm instance."""
        super().setup()
        # We defer instantiation to generate, as we need num_outputs from TaskSpec
        # print(f"Internal SAT algorithm will be instantiated in 'generate'.")
        pass # Defer instantiation

    def _instantiate_internal_alg(self, num_outputs: int):
         """Helper to instantiate the internal algorithm."""
         if self.internal_algorithm is None:
            logger.info(f"Instantiating internal SATHypothesesAlgorithm with num_outputs={num_outputs}")
            try:
                # Restore passing arguments to __init__
                # Increase beam_width significantly
                self.internal_algorithm = InternalSATHypothesesAlg(
                    beam_width=5000 # Increased beam width
                )
                # Set other attributes if needed (check SATHypothesesAlgorithm structure)
                # e.g., self.internal_algorithm.max_hypotheses = self.config.get('max_hypotheses', 2000)
                # Example: If the algorithm needs num_outputs set after initialization
                # if hasattr(self.internal_algorithm, 'num_outputs'):
                #    self.internal_algorithm.num_outputs = num_outputs
                # if hasattr(self.internal_algorithm, 'max_hypotheses'):
                #    self.internal_algorithm.max_hypotheses = self.config.get('max_hypotheses', 2000)
                # Add other necessary attribute assignments here based on InternalSATHypothesesAlg structure

            except Exception as e:
                logger.error(f"Failed to instantiate internal SATHypothesesAlgorithm: {e}", exc_info=True)
                self.internal_algorithm = None

    def generate(self, task_spec: TaskSpec, features: Dict[str, Any]) -> List[CandidateProgram]:
        """
        Uses the internal SATHypothesesAlgorithm to generate CNF hypotheses.
        --- Original Code Restored ---
        """
        logger.info(f"Running {self.strategy_id} on task: {task_spec.task_id}")

        # --- 1. Determine Output Configuration ---
        # Try to get label space from tictactoe module or fallback
        try:
            # Make sure tictactoe is importable from the project root
            from tictactoe import label_space as ttt_label_space
            self.label_space = ttt_label_space
        except ImportError:
            logger.warning("Could not import label_space from tictactoe. Using default 5 outputs.")
            self.label_space = [f"label_{i}" for i in range(DEFAULT_OUTPUT_COUNT)]
        self.num_outputs = len(self.label_space)

        # --- 2. Instantiate Internal Algorithm ---
        # Instantiate if not already done, using the determined num_outputs
        try:
            # Ensure instantiation happens if needed
            if self.internal_algorithm is None:
                 logger.info("Instantiating internal algorithm because it was None.")
                 self._instantiate_internal_alg(self.num_outputs)

        except Exception as e:
             logger.error(f"Cannot proceed without internal algorithm: {e}")
             return [] # Return empty list if setup fails

        # --- 3. Process Examples ---
        # Feed examples into the internal algorithm to build its knowledge base (base_cnf)
        logger.info(f"Processing {len(task_spec.inputs)} examples...")
        processed_count = 0
        for i, (input_data, output_data) in enumerate(zip(task_spec.inputs, task_spec.outputs)):
            try:
                board_state = input_data.get('board')
                label_str = output_data.get('label')
                if board_state is None or label_str is None:
                    logger.warning(f"Skipping example {i}: Missing 'board' or 'label'.")
                    continue

                # Convert label string to index
                try:
                    label_idx = self.label_space.index(label_str)
                except ValueError:
                     logger.warning(f"Skipping example {i}: Label '{label_str}' not in known label space.")
                     continue

                # --- Call update_history --- 
                # The internal algorithm's update_history expects board_state (str), guess (int), correct_label (int).
                # We don't have a 'guess' here, but need to pass the observation and correct label.
                # Pass correct_label as the guess, assuming it doesn't harm internal logic for learning.
                self.internal_algorithm.update_history(board_state, label_idx, label_idx)
                # ----------------------------------------

                # Optional: Log progress less frequently
                if (i + 1) % 10 == 0 or (i + 1) == len(task_spec.inputs):
                    logger.info(f"  Processed {i+1}/{len(task_spec.inputs)} examples for SAT strategy.")
                processed_count += 1

            except Exception as e:
                logger.error(f"Error processing example {i} for SAT Algorithm: {e}")
                # Decide if we should continue or raise

        logger.info(f"Finished processing {processed_count} examples.")

        # Trigger final validation if needed (depends on SATHypothesesAlgorithm internal state)
        logger.info("Triggering internal hypothesis validation...")

        # --- 4. Extract Active Hypotheses ---
        logger.info("Extracting active hypotheses...")
        active_hypotheses: List[InternalHypothesis] = []
        try:
            # Access the internal list of active hypotheses
            # Try common attribute names, may need adjustment based on internal code
            if hasattr(self.internal_algorithm, 'active_hypotheses'):
                 active_hypotheses = self.internal_algorithm.active_hypotheses
            elif hasattr(self.internal_algorithm, 'hypotheses'): # Try common alternative
                 # Filter for active ones if possible (assuming an is_active flag)
                 active_hypotheses = [h for h in self.internal_algorithm.hypotheses if getattr(h, 'is_active', True)]
            else:
                 logger.warning("Could not access 'active_hypotheses' or 'hypotheses' from internal algorithm.")
                 active_hypotheses = [] # Default to empty if access fails

            if active_hypotheses:
                 logger.info(f"Retrieved {len(active_hypotheses)} potential active hypotheses.")

        except Exception as e:
            logger.error(f"Error retrieving active hypotheses: {e}", exc_info=True)

        # --- 5. Convert to CandidateProgram ---
        candidate_programs: List[CandidateProgram] = []
        for internal_hyp in active_hypotheses:
            try:
                # Ensure clauses are in a serializable format (list of lists of ints)
                clauses = internal_hyp.clauses
                if isinstance(clauses, CNF): # If it's a CNF object
                    clauses_list = clauses.clauses
                elif isinstance(clauses, list): # Assume it's already list[list[int]]
                     clauses_list = clauses
                else:
                     logger.warning(f"Unexpected clause type {type(clauses)}. Skipping hypothesis.")
                     continue

                candidate = CandidateProgram(
                    program_representation=clauses_list, # Store as list[list[int]]
                    representation_type='CNF',
                    source_strategy=self.strategy_id,
                    confidence=getattr(internal_hyp, 'score', None), # Use score if available
                    verification_status=VerificationStatus.NOT_VERIFIED, # Verifier will handle this
                    provenance={
                        'complexity': getattr(internal_hyp, 'complexity', None),
                        # Add other relevant details from internal_hyp if needed
                        'internal_score': getattr(internal_hyp, 'score', None),
                        'internal_prob': getattr(internal_hyp, 'posterior_prob', None),
                    }
                )
                candidate_programs.append(candidate)
            except Exception as e:
                 logger.error(f"Error converting internal hypothesis to CandidateProgram: {e}", exc_info=True)

        logger.info(f"Generated {len(candidate_programs)} CandidateProgram objects.")
        return candidate_programs

    def teardown(self):
        """Clean up resources if needed."""
        super().teardown()
        self.internal_algorithm = None # Release instance
        logger.info(f"{self.strategy_id} torn down.") 