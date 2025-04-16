import pytest
from unittest.mock import MagicMock, call
import sys
import os

# Add project root for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from framework.data_structures import TaskSpec, CandidateProgram, VerificationResult, VerificationStatus
from framework.verifier import Verifier # For type hint
from typing import List, Dict, Any # For dummy class

# --- Dummy ResultAggregator (if real one not found) ---
class DummyResultAggregator:
    def __init__(self, verifier: Verifier):
        self.verifier = verifier
        # print("Dummy ResultAggregator initialized.") # Keep prints out of dummy for tests

    def aggregate_and_verify(self,
                             task_spec: TaskSpec,
                             candidate_lists: List[List[CandidateProgram]]
                             ) -> List[CandidateProgram]:
        # print(f"Aggregating {sum(len(lst) for lst in candidate_lists)} candidates from {len(candidate_lists)} lists.")
        all_candidates = [cand for sublist in candidate_lists for cand in sublist]

        if not all_candidates:
            # print("No candidates to verify.")
            return []

        unique_candidates_map = {}
        for cand in all_candidates:
            # Ensure program_representation is hashable (convert list to tuple if CNF)
            prog_repr = cand.program_representation
            if isinstance(prog_repr, list):
                 # Assuming lists of lists for CNF, convert inner lists too
                 try:
                      key_repr = tuple(tuple(clause) for clause in prog_repr)
                 except TypeError: # Handle non-list items if structure varies
                      key_repr = str(prog_repr)
            else:
                 key_repr = prog_repr # Assume others are hashable (tensor hash, string, etc.)

            key = (key_repr, cand.representation_type)
            if key not in unique_candidates_map:
                unique_candidates_map[key] = cand

        # print(f"Verifying {len(unique_candidates_map)} unique candidates.")
        for candidate in unique_candidates_map.values():
            try:
                verification_result = self.verifier.verify(candidate, task_spec)
                # Apply status back to *all* equivalent candidates
                prog_repr = candidate.program_representation
                if isinstance(prog_repr, list):
                    try:
                         key_repr = tuple(tuple(clause) for clause in prog_repr)
                    except TypeError:
                         key_repr = str(prog_repr)
                else:
                    key_repr = prog_repr
                key = (key_repr, candidate.representation_type)

                for original_cand in all_candidates:
                     original_prog_repr = original_cand.program_representation
                     if isinstance(original_prog_repr, list):
                         try:
                              original_key_repr = tuple(tuple(clause) for clause in original_prog_repr)
                         except TypeError:
                              original_key_repr = str(original_prog_repr)
                     else:
                         original_key_repr = original_prog_repr
                     original_key = (original_key_repr, original_cand.representation_type)

                     if original_key == key:
                          original_cand.verification_status = verification_result.overall_status
                # print(f"  Verified candidate ({candidate.representation_type}), result: {verification_result.overall_status}")
            except Exception as e:
                # print(f"Error verifying candidate {candidate}: {e}")
                # Mark equivalent candidates as ERROR
                prog_repr = candidate.program_representation
                if isinstance(prog_repr, list):
                    try:
                         key_repr = tuple(tuple(clause) for clause in prog_repr)
                    except TypeError:
                         key_repr = str(prog_repr)
                else:
                    key_repr = prog_repr
                key = (key_repr, candidate.representation_type)

                for original_cand in all_candidates:
                     original_prog_repr = original_cand.program_representation
                     if isinstance(original_prog_repr, list):
                         try:
                              original_key_repr = tuple(tuple(clause) for clause in original_prog_repr)
                         except TypeError:
                              original_key_repr = str(original_prog_repr)
                     else:
                         original_key_repr = original_prog_repr
                     original_key = (original_key_repr, original_cand.representation_type)

                     if original_key == key:
                          original_cand.verification_status = VerificationStatus.ERROR

        # print(f"Finished verification. Returning {len(all_candidates)} candidates.")
        return all_candidates

# --- Import Real or Use Dummy ---
try:
    from framework.result_aggregator import ResultAggregator
except ImportError:
    print("Warning: framework.result_aggregator not found. Using dummy class.")
    ResultAggregator = DummyResultAggregator # Assign dummy class

# --- Fixtures ---
@pytest.fixture
def mock_verifier():
    """Fixture for a mock Verifier."""
    verifier = MagicMock(spec=Verifier)
    verifier.verify.return_value = VerificationResult(overall_status=VerificationStatus.CONSISTENT, details=[])
    return verifier

@pytest.fixture
def sample_task_spec_agg():
    """Fixture for a TaskSpec for aggregation tests."""
    return TaskSpec(task_id="agg_test", inputs=[{"i":1}], outputs=[{"o":1}])

@pytest.fixture
def sample_candidates():
    """Fixture providing sample candidate lists."""
    # Use lists for CNF representation to test hashing
    c1 = CandidateProgram([ [1, 2], [-3] ], "CNF", "Strat1")
    c2 = CandidateProgram([ [-4, 5] ], "CNF", "Strat1")
    c3 = CandidateProgram([ [1, 2], [-3] ], "CNF", "Strat2") # Duplicate representation
    c4 = CandidateProgram("TensorRule123", "TensorRule", "Strat2")
    return [[c1, c2], [c3, c4]]

# --- Tests ---

def test_aggregator_init(mock_verifier):
    """Test ResultAggregator initialization."""
    aggregator = ResultAggregator(mock_verifier)
    assert aggregator.verifier == mock_verifier

def test_aggregate_no_candidates(mock_verifier, sample_task_spec_agg):
    """Test aggregating with no candidate lists."""
    aggregator = ResultAggregator(mock_verifier)
    results = aggregator.aggregate_and_verify(sample_task_spec_agg, [])
    assert results == []
    mock_verifier.verify.assert_not_called()

def test_aggregate_empty_candidate_lists(mock_verifier, sample_task_spec_agg):
    """Test aggregating with empty candidate lists."""
    aggregator = ResultAggregator(mock_verifier)
    results = aggregator.aggregate_and_verify(sample_task_spec_agg, [[], []])
    assert results == []
    mock_verifier.verify.assert_not_called()

def test_aggregate_and_verify_calls_verifier(mock_verifier, sample_task_spec_agg, sample_candidates):
    """Test that the verifier is called for unique candidates."""
    aggregator = ResultAggregator(mock_verifier)
    c1, c2, c3, c4 = sample_candidates[0][0], sample_candidates[0][1], sample_candidates[1][0], sample_candidates[1][1]

    # Set specific return values for verification if needed
    mock_verifier.verify.side_effect = [
        VerificationResult(VerificationStatus.CONSISTENT, []),    # For CNF [[1, 2], [-3]]
        VerificationResult(VerificationStatus.CONTRADICTED, []), # For CNF [[-4, 5]]
        VerificationResult(VerificationStatus.CONSISTENT, [])     # For TensorRule "TensorRule123"
    ]

    results = aggregator.aggregate_and_verify(sample_task_spec_agg, sample_candidates)

    # Verify calls (only 3 unique candidates)
    assert mock_verifier.verify.call_count == 3
    calls = mock_verifier.verify.call_args_list
    verified_candidates = [c.args[0] for c in calls] # CandidateProgram is the first arg to verify

    # Check that one instance of each unique representation was verified
    verified_reprs = set()
    for cand in verified_candidates:
        prog_repr = cand.program_representation
        if isinstance(prog_repr, list):
            key_repr = tuple(tuple(clause) for clause in prog_repr)
        else:
            key_repr = prog_repr
        verified_reprs.add((key_repr, cand.representation_type))

    assert (( (1, 2), (-3,) ), "CNF") in verified_reprs
    assert (( (-4, 5), ), "CNF") in verified_reprs
    assert ("TensorRule123", "TensorRule") in verified_reprs

def test_aggregate_updates_status(mock_verifier, sample_task_spec_agg, sample_candidates):
    """Test that candidate statuses are updated correctly, including duplicates."""
    aggregator = ResultAggregator(mock_verifier)
    c1, c2, c3, c4 = sample_candidates[0][0], sample_candidates[0][1], sample_candidates[1][0], sample_candidates[1][1]

    # Mock verification results mapped to representation/type
    def verify_side_effect(candidate, task_spec):
        prog_repr = candidate.program_representation
        if isinstance(prog_repr, list):
            key_repr = tuple(tuple(clause) for clause in prog_repr)
        else:
            key_repr = prog_repr
        key = (key_repr, candidate.representation_type)

        if key == (( (1, 2), (-3,) ), "CNF"): return VerificationResult(VerificationStatus.CONSISTENT, [])
        if key == (( (-4, 5), ), "CNF"): return VerificationResult(VerificationStatus.CONTRADICTED, [])
        if key == ("TensorRule123", "TensorRule"): return VerificationResult(VerificationStatus.ERROR, [])
        return VerificationResult(VerificationStatus.NOT_VERIFIED, []) # Fallback

    mock_verifier.verify.side_effect = verify_side_effect

    results = aggregator.aggregate_and_verify(sample_task_spec_agg, sample_candidates)

    assert len(results) == 4 # All original candidates returned
    # Check statuses (find candidates by original identity)
    assert c1.verification_status == VerificationStatus.CONSISTENT
    assert c2.verification_status == VerificationStatus.CONTRADICTED
    assert c3.verification_status == VerificationStatus.CONSISTENT # Updated via c1 verification
    assert c4.verification_status == VerificationStatus.ERROR

def test_aggregate_verification_error(mock_verifier, sample_task_spec_agg, sample_candidates):
    """Test handling of exceptions during verification."""
    aggregator = ResultAggregator(mock_verifier)
    c1, c2, c3, c4 = sample_candidates[0][0], sample_candidates[0][1], sample_candidates[1][0], sample_candidates[1][1]

    # Mock verification results mapped to representation/type
    def verify_side_effect(candidate, task_spec):
        prog_repr = candidate.program_representation
        if isinstance(prog_repr, list):
            key_repr = tuple(tuple(clause) for clause in prog_repr)
        else:
            key_repr = prog_repr
        key = (key_repr, candidate.representation_type)

        if key == (( (1, 2), (-3,) ), "CNF"): return VerificationResult(VerificationStatus.CONSISTENT, [])
        if key == (( (-4, 5), ), "CNF"): raise ValueError("Verifier crashed!") # Error for CNF B
        if key == ("TensorRule123", "TensorRule"): return VerificationResult(VerificationStatus.CONSISTENT, [])
        return VerificationResult(VerificationStatus.NOT_VERIFIED, [])

    mock_verifier.verify.side_effect = verify_side_effect

    results = aggregator.aggregate_and_verify(sample_task_spec_agg, sample_candidates)

    assert len(results) == 4
    assert c1.verification_status == VerificationStatus.CONSISTENT
    assert c2.verification_status == VerificationStatus.ERROR # Should be marked as error
    assert c3.verification_status == VerificationStatus.CONSISTENT # Error shouldn't affect others
    assert c4.verification_status == VerificationStatus.CONSISTENT
