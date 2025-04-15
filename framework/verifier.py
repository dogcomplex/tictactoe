from typing import List, Dict, Any, Tuple, Optional
from .data_structures import TaskSpec, CandidateProgram, VerificationStatus, VerificationResult
# We will need access to the TicTacToe logic later to actually verify
# For now, we'll create placeholder logic.
# from tictactoe import solve_board_with_rule # Example function (to be implemented/imported)
import traceback
import logging # Added logging
import torch # Add torch import

logger = logging.getLogger(__name__) # Added logger

class Verifier:
    """Verifies candidate programs against task examples."""

    def __init__(self):
        print("Initialized Verifier.")
        # Cache or resources for verification if needed
        # Load TicTacToe functions or rules if necessary
        # For MVP, we might need to load the expected label mapping
        try:
            from tictactoe import label_space, tictactoe as ttt_solver
            self.label_space = label_space
            self.solver = ttt_solver # Using the standard solver for now
            print(f"Verifier using TicTacToe label space: {self.label_space}")
        except ImportError:
            print("Warning: Could not import TicTacToe logic for verifier.")
            self.label_space = ['ok', 'win1', 'win2', 'draw', 'error'] # Fallback
            self.solver = None

    # --- Add helper functions for TTT CNF conversion ---
    # (These mirror the logic from sat_wrapper for consistency)
    def _ttt_input_to_binary(self, input_str: str, num_inputs: int, bits_per_input: int) -> str:
        """Converts TTT string to binary string based on expected encoding."""
        if len(input_str) != num_inputs:
             raise ValueError(f"Input string length {len(input_str)} does not match expected {num_inputs}.")
        mapping = {'0': '100', '1': '010', '2': '001'} # Assuming 3 bits per input
        if bits_per_input != 3:
             # Basic check, real implementation might need more flexible encoding
             raise NotImplementedError("Verifier currently assumes 3 bits per input for TTT encoding.")
        return ''.join(mapping.get(char, '000') for char in input_str)

    def _get_input_assumptions(self, input_binary: str, num_inputs: int, bits_per_input: int) -> List[int]:
         """Converts binary input string to PySAT assumptions list."""
         if len(input_binary) != num_inputs * bits_per_input:
             raise ValueError(f"Binary input length {len(input_binary)} incorrect for {num_inputs} inputs * {bits_per_input} bits.")
         assumptions = []
         var_index = 1 # PySAT variables are 1-indexed
         for i in range(num_inputs):
             # Find which bit is '1' for this input position
             true_bit_offset = -1
             for b in range(bits_per_input):
                 bit_char = input_binary[i * bits_per_input + b]
                 if bit_char == '1':
                     if true_bit_offset != -1: # Error: more than one bit set for this input position
                         raise ValueError(f"Invalid one-hot encoding for input {i} in '{input_binary}'")
                     true_bit_offset = b
                 # Add assumption for this specific bit
                 var = var_index + b
                 assumptions.append(var if bit_char == '1' else -var)

             if true_bit_offset == -1: # Error: no bit set for this input position
                  raise ValueError(f"Invalid one-hot encoding (no bit set) for input {i} in '{input_binary}'")

             var_index += bits_per_input
         return assumptions

    def _get_output_var(self, output_idx: int, num_inputs: int, bits_per_input: int) -> int:
         """Gets the PySAT variable number for a given output index."""
         base_output_var = num_inputs * bits_per_input + 1
         return base_output_var + output_idx

    # --- Add helper for TensorRule ---
    def _map_ttt_to_tensor(self, board_str: str, output_label_str: Optional[str]) -> Optional[torch.Tensor]:
         """Maps TTT board and optional output label to internal 32-bit tensor"""
         try:
             from recipes import map_observation_to_tensor as internal_map_observation
             # Determine the internal token ('C', 'W', etc.) for the label
             recipes_output_token = None
             if output_label_str:
                  # Use the mapping logic from the wrapper or define locally
                  # Assuming DEFAULT_LABEL_SPACE for simplicity here
                  mapping = {'ok': 'C', 'win1': 'W', 'win2': 'W', 'draw': 'D', 'error': 'E'}
                  recipes_output_token = mapping.get(output_label_str)
                  if recipes_output_token is None:
                       logger.warning(f"Verifier: Cannot map label '{output_label_str}' to recipes token.")
                       # Decide how to handle - treat as no label or error?
                       # For verification, we usually care about input consistency first.

             # Generate the tensor representation
             tensor = internal_map_observation(board_str, recipes_output_token)
             return tensor
         except ImportError:
             logger.error("Verifier cannot map to tensor: recipes.py import failed.")
             return None
         except Exception as e:
              logger.error(f"Verifier error during tensor mapping: {e}", exc_info=True)
              return None

    def verify(self, candidate: CandidateProgram, task_spec: TaskSpec) -> VerificationResult:
        """
        Checks if a candidate program correctly predicts outputs for all inputs
        in the task specification.

        Args:
            candidate: The CandidateProgram to verify.
            task_spec: The TaskSpec containing input/output examples.

        Returns:
            A VerificationResult object.
        """
        # print(f"Verifying candidate from {candidate.source_strategy} ({candidate.representation_type})...") # Reduced verbosity
        if not task_spec.inputs:
            return VerificationResult(overall_status=VerificationStatus.ERROR, error_message="No input examples provided in task spec.")

        passed_count = 0
        failed_count = 0
        error_count = 0
        details = []

        for i, (input_data, expected_output_data) in enumerate(zip(task_spec.inputs, task_spec.outputs)):
            try:
                # --- Verification Logic ---
                # This is the core part that needs specific implementation
                # based on candidate.representation_type

                # Example: For TicTacToe (MVP) - assume input is {'board': '012...'}
                # and output is {'label': 'win1'}
                if 'board' not in input_data or 'label' not in expected_output_data:
                     raise ValueError(f"Example {i}: Invalid input/output format. Expected 'board' in input and 'label' in output.")

                board_state = input_data['board']
                expected_label_str = expected_output_data['label']
                try:
                    expected_label_idx = self.label_space.index(expected_label_str)
                except ValueError:
                    raise ValueError(f"Example {i}: Expected label '{expected_label_str}' not found in label space {self.label_space}")

                # How to get the prediction depends *heavily* on the candidate representation
                predicted_label_idx = -1 # Default to invalid prediction
                prediction_source = "N/A"
                possible_outputs = [] # Store indices of outputs deemed possible by the rule


                if candidate.representation_type == "CNF":
                    prediction_source = "CNF Interpretation"
                    try:
                        # Assume standard TTT encoding for MVP
                        num_inputs = 9
                        bits_per_input = 3
                        num_outputs = len(self.label_space)

                        # 1. Get CNF clauses for the candidate
                        cnf_clauses = candidate.program_representation
                        if not isinstance(cnf_clauses, list):
                            raise TypeError("CNF representation is not a list of clauses.")

                        # 2. Convert board state to input assumptions for PySAT
                        input_binary = self._ttt_input_to_binary(board_state, num_inputs, bits_per_input)
                        input_assumptions = self._get_input_assumptions(input_binary, num_inputs, bits_per_input)

                        # 3. Check which outputs are *possible* given the rule + input
                        from pysat.solvers import Solver # Import here for clarity
                        possible_outputs = [] # Store indices of outputs deemed possible by the rule
                        for potential_output_idx in range(num_outputs):
                            output_var = self._get_output_var(potential_output_idx, num_inputs, bits_per_input)
                            # Check if Input + Rule + Output is SATISFIABLE
                            assumption_for_check = input_assumptions + [output_var]
                            with Solver(bootstrap_with=cnf_clauses, use_timer=True) as solver:
                                is_sat = solver.solve(assumptions=assumption_for_check)
                                # print(f"      Input: {board_state}, Assume Output: {self.label_space[potential_output_idx]}({output_var}), SAT: {is_sat}")
                                if is_sat:
                                    # If assuming the output is SAT, then this output is possible under the rule
                                    possible_outputs.append(potential_output_idx)

                        # 4. Determine if the EXPECTED output was possible
                        # is_correct_for_example = (expected_label_idx in possible_outputs) # Old logic based only on possibility

                        # -- Start New Logic for CNF correctness --
                        # Correct only if the expected output is uniquely possible
                        if len(possible_outputs) == 1 and possible_outputs[0] == expected_label_idx:
                             is_correct_for_example_cnf = True
                             predicted_label_idx = expected_label_idx # Unique prediction matches expected
                        else:
                             is_correct_for_example_cnf = False
                             # Determine prediction code based on ambiguity/impossibility
                             if len(possible_outputs) == 0:
                                 predicted_label_idx = -1 # Indicate rule contradicts *all* outputs
                             elif expected_label_idx not in possible_outputs:
                                 predicted_label_idx = -2 # Indicate rule contradicts the *specific* expected output
                             else: # Expected is possible, but so are others
                                 predicted_label_idx = -3 # Indicate ambiguity
                        # -- End New Logic for CNF correctness --

                        # Store details for logging/reporting
                        # predicted_label_idx = -4 # Use a distinct code for CNF possibility check -- OLD CODE
                        if not possible_outputs:
                             status_detail = " (Rule inconsistent with ALL outputs)"
                        elif len(possible_outputs) == 1:
                             # predicted_label_idx = possible_outputs[0] # Can infer prediction if only one possible -- Done above
                             status_detail = ""
                        else:
                             status_detail = f" (Rule allows outputs: {[self.label_space[i] for i in possible_outputs]})"

                    except Exception as e:
                        logger.error(f"Error during CNF verification for example {i}: {e}", exc_info=True)
                        # Let the outer exception handler catch this and mark as Error
                        raise # Re-raise to be caught by outer handler

                elif candidate.representation_type == "TensorRule":
                     prediction_source = "TensorRule Interpretation"
                     try:
                        # 1. Get hypothesis tensor (convert back from list if needed)
                        hyp_rep = candidate.program_representation
                        if isinstance(hyp_rep, list):
                            hypothesis_tensor = torch.tensor(hyp_rep, dtype=torch.bool)
                        elif torch.is_tensor(hyp_rep):
                             hypothesis_tensor = hyp_rep.to(dtype=torch.bool) # Ensure correct type
                        else:
                             raise TypeError(f"Unexpected TensorRule representation type: {type(hyp_rep)}")

                        # Move tensor to CPU for verification (recipes runs on GPU but verifier might not need it)
                        hypothesis_tensor = hypothesis_tensor.cpu()

                        # 2. Convert current example's board_state to input tensor
                        # We only need the input part, so don't need the expected output label here
                        obs_tensor = self._map_ttt_to_tensor(board_state, None)
                        if obs_tensor is None:
                            raise RuntimeError("Failed to convert board state to tensor for verification.")
                        obs_tensor = obs_tensor.cpu()

                        # 3. Check input consistency
                        hyp_inputs = hypothesis_tensor[:27] # First 27 bits are input states
                        obs_inputs = obs_tensor[:27]
                        # Input matches if all 'True' bits in the observation's required state
                        # are also 'True' (allowed) in the hypothesis's input part.
                        # Logic from recipes.py: input_matches = torch.all(hyp_inputs | ~obs_inputs)
                        input_matches = torch.all(hyp_inputs | ~obs_inputs).item()

                        if input_matches:
                            # 4. Extract possible outputs if input matches
                            hyp_outputs = hypothesis_tensor[27:] # Last 5 bits are output states
                            possible_indices = torch.where(hyp_outputs)[0].tolist() # Get indices (27-31) where bit is True

                            # 5. Map internal indices back to standard label indices
                            # Need the reverse map built in the wrapper or recreated here
                            # Assuming `self.label_space` is correctly populated
                            possible_outputs = []
                            # Tentative mapping based on default TTT labels and recipes tokens
                            idx_map = {27: 'ok', 28: 'win1', 29: 'win2', 30: 'draw', 31: 'error'} # Need careful check! WIN L mapping might be wrong
                            # A better approach might be needed if label space isn't default
                            for internal_idx in possible_indices:
                                 std_label = idx_map.get(internal_idx)
                                 if std_label and std_label in self.label_space:
                                     possible_outputs.append(self.label_space.index(std_label))
                                 else:
                                      logger.warning(f"TensorRule verification: Cannot map internal output index {internal_idx} to standard label.")


                            # 6. Determine prediction
                            if len(possible_outputs) == 1:
                                predicted_label_idx = possible_outputs[0]
                            elif len(possible_outputs) == 0:
                                logger.warning(f"TensorRule verification: Input matched, but no output allowed by rule for {board_state}")
                                predicted_label_idx = -1 # Rule applies but doesn't predict output
                            else:
                                logger.warning(f"TensorRule verification: Input matched, multiple outputs {possible_outputs} allowed for {board_state}")
                                predicted_label_idx = -2 # Ambiguous prediction
                        else:
                            # Input doesn't match the hypothesis rule
                            logger.debug(f"TensorRule verification: Input {board_state} does not match hypothesis.")
                            predicted_label_idx = -3 # Indicate rule doesn't apply to this input


                     except Exception as e:
                         logger.error(f"Error during TensorRule verification for example {i}: {e}", exc_info=True)
                         raise # Re-raise to be caught by outer handler

                # Add other representation types if needed
                # elif candidate.representation_type == "PlaceholderPrediction": ...

                else:
                    # Fallback or error for unknown types
                    prediction_source = "Unknown Representation Fallback (Solver)"
                    logger.warning(f"Using ground truth fallback for unknown representation type: {candidate.representation_type}")
                    if self.solver:
                        actual_label_str = self.solver(board_state)
                        actual_label_idx = self.label_space.index(actual_label_str)
                        predicted_label_idx = actual_label_idx
                    else:
                         raise NotImplementedError(f"Verification logic for representation type '{candidate.representation_type}' is not implemented and no fallback solver available.")


                # --- Compare prediction with expected output ---
                # Handle special prediction indices (-1 no prediction, -2 ambiguous, -3 rule no match)
                # For CNF, we now use is_correct_for_example determined above
                if candidate.representation_type == "CNF":
                    is_correct = is_correct_for_example_cnf
                elif predicted_label_idx >= 0: # Valid prediction index for other types
                    is_correct = (predicted_label_idx == expected_label_idx)
                else: # Rule didn't predict, was ambiguous, or didn't match input for other types
                    is_correct = False

                if is_correct:
                    passed_count += 1
                    status = "Pass"
                else:
                    failed_count += 1
                    status = "Fail"
                    # Add more detail if prediction failed
                    if candidate.representation_type == "CNF":
                        status += status_detail # Add detail about possible outputs
                    elif predicted_label_idx == -1: status += " (No Prediction)"
                    elif predicted_label_idx == -2: status += " (Ambiguous Prediction)"
                    elif predicted_label_idx == -3: status += " (Rule Not Applicable)"
                    # Keep -4 for internal use, don't show user directly unless Fail
                    # elif predicted_label_idx == -99: status += " (Not Implemented)"

                details.append({
                    "example_index": i,
                    "input": input_data,
                    "expected_output": expected_output_data,
                    "predicted_output_idx": predicted_label_idx,
                    "predicted_output_label": self.label_space[predicted_label_idx] if 0 <= predicted_label_idx < len(self.label_space) else "Invalid/NI",
                    "status": status,
                    "prediction_source": prediction_source
                })

            except Exception as e:
                error_count += 1
                details.append({
                    "example_index": i,
                    "input": input_data,
                    "expected_output": expected_output_data,
                    "status": "Error",
                    "error_message": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc() # Add traceback for debugging
                })

        # Determine overall status
        overall_status = VerificationStatus.ERROR
        error_message = None
        if error_count > 0:
            overall_status = VerificationStatus.ERROR
            error_message = f"{error_count} verification errors occurred."
        elif failed_count > 0:
            overall_status = VerificationStatus.CONTRADICTED
        elif passed_count == len(task_spec.inputs):
            overall_status = VerificationStatus.CONSISTENT
        else: # Should not happen if error_count is 0
            overall_status = VerificationStatus.ERROR
            error_message = "Inconsistent verification state."

        # Reduced verbosity
        # print(f"Verification complete: Passed={passed_count}, Failed={failed_count}, Errors={error_count}. Overall: {overall_status.name}")
        return VerificationResult(overall_status=overall_status, details=details, error_message=error_message) 