import torch
from typing import List, Dict, Any, Optional
import logging
import sys
import os
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from framework.strategy_interface import Strategy
from framework.data_structures import TaskSpec, CandidateProgram, VerificationStatus

# Attempt to import from recipes.py
try:
    from recipes import HypothesisManager as InternalHypothesisManager
    from recipes import map_observation_to_tensor as internal_map_observation
    from recipes import index_to_token as internal_index_to_token # Get the internal token mapping
    recipes_num_tokens = len(internal_index_to_token) # Should be 32
except ImportError as e:
    print(f"Error importing from recipes.py: {e}. Ensure recipes.py is in the project root.")
    # Dummy classes for parsing
    class InternalHypothesisManager:
         def __init__(self, device, use_disk_cache=False): pass
         def update_valid_hypotheses(self, obs_tensor: torch.Tensor): pass
         def get_top_hypotheses(self, n=10, observation_tensors=None, matching_indices=None, include_invalid=False, include_miss=False): return [], []
         def visualize_hypothesis(self, hypothesis): return "Dummy Hypothesis"
    def internal_map_observation(board, output_char): return torch.zeros(32, dtype=torch.bool) # Dummy tensor
    internal_index_to_token = {}
    recipes_num_tokens = 32

logger = logging.getLogger(__name__)

# --- Mapping between standard labels and recipes.py internal tokens ---
# recipes.py uses 'C', 'W', 'L', 'D', 'E' at indices 27-31
# Need to map from standard ['ok', 'win1', 'win2', 'draw', 'error']
DEFAULT_LABEL_SPACE = ['ok', 'win1', 'win2', 'draw', 'error']
# This mapping needs careful verification based on recipes.py logic meaning
# Assuming 'C' corresponds to 'ok', 'W' to 'win1'/'win2', 'L' maybe unused?, 'D' to 'draw', 'E' to 'error'
# For simplicity in MVP, let's do a direct mapping if lengths match, otherwise error.
# We might need a more robust mapping later.
RECIPES_OUTPUT_TOKENS = {v: k for k, v in internal_index_to_token.items() if k >= 27} # {27: 'C', 28: 'W', ...}

def map_standard_label_to_recipes_token(label: str, label_space: List[str]) -> Optional[str]:
    """Maps standard label string to recipes.py output token ('C', 'W', etc.)."""
    if label_space == DEFAULT_LABEL_SPACE:
         # Tentative mapping based on assumptions
         mapping = {
             'ok': 'C',
             'win1': 'W', # Needs care - recipes uses 'W' for general win?
             'win2': 'W', # Needs care
             'draw': 'D',
             'error': 'E'
         }
         return mapping.get(label)
    else:
        # If not the default space, we can't assume the mapping
        logger.warning(f"Cannot map label '{label}': unexpected label_space {label_space}")
        return None


class RecipesStrategyWrapper(Strategy):
    """
    Wrapper for the recipes.py HypothesisManager system.
    """
    strategy_id: str = "Recipes_HypothesisManager_v1"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.internal_manager: Optional[InternalHypothesisManager] = None
        self.device = None
        self.label_space: List[str] = []
        self.recipes_label_map: Dict[int, str] = {} # Maps internal token index (27-31) to standard label

        # --- Configuration ---
        self.use_disk_cache = self.config.get('use_disk_cache', False)
        self.top_n_hypotheses = self.config.get('top_n_hypotheses', 50) # How many hypotheses to retrieve

    def setup(self):
        """Initialize the internal HypothesisManager."""
        super().setup()
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"{self.strategy_id} using device: {self.device}")
            if recipes_num_tokens != 32:
                 raise RuntimeError(f"recipes.py num_tokens ({recipes_num_tokens}) is not 32. Incompatible.")

            self.internal_manager = InternalHypothesisManager(
                device=self.device,
                use_disk_cache=self.use_disk_cache
            )
            # Initialization of the manager might take time (generating/loading hypotheses)
            logger.info(f"Internal HypothesisManager initialized (use_disk_cache={self.use_disk_cache}).")

        except Exception as e:
            logger.error(f"Failed to setup internal HypothesisManager: {e}", exc_info=True)
            self.internal_manager = None # Ensure it's None if setup fails

    def _build_recipes_label_map(self):
        """Build the mapping from internal output token indices to standard labels."""
        self.recipes_label_map = {}
        if not self.label_space: return # Cannot map without standard labels

        # Based on DEFAULT_LABEL_SPACE assumptions
        if self.label_space == DEFAULT_LABEL_SPACE:
            internal_token_map = {token: idx for idx, token in RECIPES_OUTPUT_TOKENS.items()} # {'C': 27, 'W': 28,...}
            label_to_token = map_standard_label_to_recipes_token # Use the helper
            for std_label in self.label_space:
                recipes_token = label_to_token(std_label, self.label_space)
                if recipes_token and recipes_token in internal_token_map:
                     # Store index -> standard label
                     # Handle 'W' mapping to both 'win1' and 'win2' - prioritize first? Needs clarification.
                     idx = internal_token_map[recipes_token]
                     if idx not in self.recipes_label_map: # Avoid overwriting 'W' mapping
                        self.recipes_label_map[idx] = std_label
                     elif std_label.startswith('win'): # Allow 'W' to map to 'win1' or 'win2' if first was already set
                         # This isn't ideal, ambiguity remains if 'W' is predicted
                         pass
        else:
             logger.warning(f"Cannot automatically map recipes labels for non-default label space: {self.label_space}")


    def generate(self, task_spec: TaskSpec, features: Dict[str, Any]) -> List[CandidateProgram]:
        """
        Uses the internal HypothesisManager to process examples and return valid hypotheses.
        """
        logger.info(f"Running {self.strategy_id} on task: {task_spec.task_id}")
        start_time = time.time()

        if self.internal_manager is None:
            logger.error(f"{self.strategy_id} setup failed or was not called. Cannot generate.")
            return []

        # --- 1. Determine Output Configuration & Mapping ---
        try:
            from tictactoe import label_space as ttt_label_space
            self.label_space = ttt_label_space
        except ImportError:
            logger.warning("Could not import label_space from tictactoe. Using default.")
            self.label_space = DEFAULT_LABEL_SPACE
        self._build_recipes_label_map()
        logger.debug(f"Using label space: {self.label_space}")
        logger.debug(f"Recipes token index to standard label map: {self.recipes_label_map}")


        # --- 2. Process Examples (Update Valid Hypotheses) ---
        logger.info(f"Processing {len(task_spec.inputs)} examples to filter hypotheses...")
        processed_count = 0
        for i, (input_data, output_data) in enumerate(zip(task_spec.inputs, task_spec.outputs)):
            try:
                if 'board' not in input_data or 'label' not in output_data:
                    logger.warning(f"Skipping example {i}: Missing 'board' or 'label'.")
                    continue

                board_str = input_data['board']
                label_str = output_data['label']

                # Map standard label to internal recipes token ('C', 'W', etc.)
                recipes_output_token = map_standard_label_to_recipes_token(label_str, self.label_space)
                if recipes_output_token is None:
                    logger.warning(f"Skipping example {i}: Could not map label '{label_str}' to recipes token.")
                    continue

                # Convert observation to internal tensor format
                # Need to pass the *internal* token ('C', 'W', etc.)
                obs_tensor = internal_map_observation(board_str, recipes_output_token)
                obs_tensor = obs_tensor.to(self.device) # Move to correct device

                # Update the manager's valid hypothesis set
                # Pass the correct label string as expected by the internal method
                self.internal_manager.update_valid_hypotheses(obs_tensor, label_str)
                processed_count += 1

                if (processed_count % 50 == 0) or (processed_count == len(task_spec.inputs)):
                     logger.info(f"Processed {processed_count}/{len(task_spec.inputs)} examples...")


            except Exception as e:
                logger.error(f"Error processing example {i} for HypothesisManager: {e}", exc_info=True)

        logger.info(f"Finished processing {processed_count} examples for filtering.")
        if processed_count == 0 and len(task_spec.inputs) > 0:
             logger.warning("No examples were successfully processed for filtering. Hypotheses may be unconstrained.")

        # --- Check if any valid hypotheses remain ---
        num_valid_hypotheses = torch.sum(self.internal_manager.valid_hypotheses).item()
        if num_valid_hypotheses == 0:
            logger.info("No valid hypotheses remaining after filtering. Returning empty list.")
            return [] # Return empty list if no hypotheses are valid
        # --- End Check --- 

        logger.info(f"Extracting top {self.top_n_hypotheses} valid hypotheses...")
        # valid_hypotheses_indices = [] # Indices might not be returned
        valid_hypotheses_tensors = []
        try:
            # get_top_hypotheses likely returns just the tensors based on the error
            # Adjust the call and assignment
            hypotheses_result = self.internal_manager.get_top_hypotheses(
                n=self.top_n_hypotheses,
                include_invalid=False, # Only get hypotheses consistent with processed examples
                include_miss=True # Include those that didn't apply to any example? Maybe False. Let's try True.
            )
            # Assume it returns a list of tensors
            if isinstance(hypotheses_result, list):
                 valid_hypotheses_tensors = hypotheses_result
            else:
                 # Handle unexpected return type if necessary
                 logger.warning(f"Unexpected return type from get_top_hypotheses: {type(hypotheses_result)}")
                 # Attempt to extract tensors if it's a tuple/dict? Requires inspection.

            logger.info(f"Retrieved {len(valid_hypotheses_tensors)} valid hypothesis tensors.")
        except Exception as e:
             logger.error(f"Error retrieving hypotheses from HypothesisManager: {e}", exc_info=True)


        # --- 4. Convert to CandidateProgram ---
        candidate_programs: List[CandidateProgram] = []
        # Assume hypotheses_result is a list of tuples, maybe (index, tensor)?
        for i, hypothesis_data in enumerate(valid_hypotheses_tensors):
            try:
                # --- Unpack the actual tensor --- START
                hypothesis_tensor = None
                internal_index = i # Default if no index found
                if torch.is_tensor(hypothesis_data):
                    hypothesis_tensor = hypothesis_data
                elif isinstance(hypothesis_data, tuple) and len(hypothesis_data) > 0 and torch.is_tensor(hypothesis_data[0]):
                    # Assuming tensor is the first element if it's a tuple
                    hypothesis_tensor = hypothesis_data[0]
                    # Try to get index if it's the second element
                    if len(hypothesis_data) > 1 and isinstance(hypothesis_data[1], int):
                         internal_index = hypothesis_data[1]
                else:
                     logger.warning(f"Could not extract tensor from hypothesis data item {i} (type: {type(hypothesis_data)}). Skipping.")
                     continue # Skip this item
                # --- Unpack the actual tensor --- END

                # Representation: Store the boolean tensor itself (or convert to list for JSON)
                # Convert tensor to list[bool] for easier serialization/handling if needed
                representation = hypothesis_tensor.cpu().tolist() # Convert to list on CPU

                # Try to add a human-readable version to provenance
                readable_rep = "N/A"
                try:
                    # Pass the actual tensor to visualize
                    readable_rep = self.internal_manager.visualize_hypothesis(hypothesis_tensor)
                except Exception as viz_e:
                    logger.warning(f"Could not visualize hypothesis {i}: {viz_e}")


                candidate = CandidateProgram(
                    program_representation=representation,
                    representation_type="TensorRule", # Indicate it's a tensor from recipes
                    source_strategy=self.strategy_id,
                    confidence=None, # recipes.py doesn't seem to have explicit confidence
                    verification_status=VerificationStatus.NOT_VERIFIED,
                    provenance={
                        'internal_index': internal_index, # Keep track of original index if available
                        'readable_form': readable_rep,
                        # Add stats if they were returned by get_top_hypotheses
                        'stats': hypothesis_data[2] if isinstance(hypothesis_data, tuple) and len(hypothesis_data) > 2 else None
                    }
                )
                candidate_programs.append(candidate)
            except Exception as e:
                logger.error(f"Error converting hypothesis {i} to CandidateProgram: {e}", exc_info=True)

        duration = time.time() - start_time
        logger.info(f"{self.strategy_id} finished in {duration:.2f}s. Generated {len(candidate_programs)} candidates.")
        return candidate_programs

    def teardown(self):
        """Clean up resources, if any."""
        super().teardown()
        # Check if internal_manager exists and has a teardown method
        if hasattr(self.internal_manager, 'teardown') and callable(getattr(self.internal_manager, 'teardown')):
            try:
                logger.info(f"Calling teardown on internal {type(self.internal_manager).__name__}...")
                self.internal_manager.teardown()
            except Exception as e:
                logger.error(f"Error during internal manager teardown: {e}", exc_info=True)
        self.internal_manager = None # Release instance
        logger.info(f"{self.strategy_id} torn down.") 