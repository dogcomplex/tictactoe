import pytest
import json
import os
from unittest.mock import patch, mock_open, MagicMock, call, ANY
import sys
import time
import datetime
import logging

# Add project root for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from framework.data_structures import TaskSpec, CandidateProgram, VerificationResult, VerificationStatus
from framework.reporter import Reporter, ResultAggregator
from framework.verifier import Verifier

# Assume Reporter is in framework.reporter
try:
    from framework.reporter import Reporter
except ImportError:
    print("Warning: framework.reporter not found. Using dummy class for tests.")
    from typing import List, Dict, Any
    logger = logging.getLogger(__name__)
    # Dummy Reporter
    class Reporter:
        def __init__(self, output_dir: str = "results"):
            self.output_dir = output_dir
            # os.makedirs(self.output_dir, exist_ok=True) # Avoid side effects in dummy
            logger.info(f"Dummy Reporter initialized. Output dir: {self.output_dir}")

        def generate_report(self, task_spec: TaskSpec, final_candidates: List[CandidateProgram], run_summary: Dict[str, Any]):
            report_data = {
                "task_id": task_spec.task_id,
                "run_summary": run_summary,
                "candidates": []
            }
            consistent_count = 0
            for cand in final_candidates:
                report_data["candidates"].append({
                    "source_strategy": cand.source_strategy,
                    "representation_type": cand.representation_type,
                    "representation": str(cand.program_representation), # Ensure serializable
                    "verification_status": cand.verification_status.name,
                    "confidence": cand.confidence,
                    "provenance": cand.provenance
                })
                if cand.verification_status == VerificationStatus.CONSISTENT:
                    consistent_count += 1

            # Log summary
            logger.info("--- Run Report ---")
            logger.info(f"Task ID: {task_spec.task_id}")
            logger.info(f"Total Candidates Generated: {len(final_candidates)}")
            logger.info(f"Consistent Candidates Found: {consistent_count}")
            logger.info(f"Run Summary: {run_summary}")
            logger.info("------------------")

            # Simulate JSON output
            json_path = os.path.join(self.output_dir, f"{task_spec.task_id}_report.json")
            # print(f"(Dummy) Writing JSON report to: {json_path}")
            # In real code: with open(json_path, 'w') as f: json.dump(report_data, f, indent=2)

# --- Fixtures ---
@pytest.fixture
def mock_candidate_program():
    """Provides a generic mock CandidateProgram."""
    mock_prog = MagicMock(spec=CandidateProgram)
    mock_prog.program_representation = "Mock Program"
    mock_prog.representation_type = "MockType"
    mock_prog.source_strategy = "MockStrategy"
    mock_prog.confidence = 0.9
    mock_prog.verification_status = VerificationStatus.NOT_VERIFIED
    mock_prog.provenance = {'origin': 'test'}
    mock_prog.verification_results = None # Added for completeness
    # Removed .to_dict access as CandidateProgram is a dataclass
    return mock_prog

@pytest.fixture
def mock_candidate_consistent(mock_candidate_program):
    # Just set the attribute directly
    mock_candidate_program.verification_status = VerificationStatus.CONSISTENT
    return mock_candidate_program

@pytest.fixture
def mock_candidate_contradicted(mock_candidate_program):
    # Just set the attribute directly
    mock_candidate_program.verification_status = VerificationStatus.CONTRADICTED
    return mock_candidate_program

@pytest.fixture
def mock_candidate_error(mock_candidate_program):
    # Just set the attribute directly
    mock_candidate_program.verification_status = VerificationStatus.ERROR
    mock_candidate_program.verification_results = {'error': 'Mock solver timed out'} # Add mock error detail
    return mock_candidate_program

@pytest.fixture
def mock_candidates_list():
    """Provides a list of distinct mock CandidatePrograms with various statuses."""
    # Create distinct mock objects for each entry
    c1 = MagicMock(spec=CandidateProgram)
    c1.program_representation = "Consistent Rep 1"
    c1.representation_type = "MockType"
    c1.source_strategy = "MockStrategy"
    c1.confidence = 0.95
    c1.verification_status = VerificationStatus.CONSISTENT
    c1.provenance = {'origin': 'test1'}
    c1.verification_results = None

    c2 = MagicMock(spec=CandidateProgram)
    c2.program_representation = "Contradicted Rep"
    c2.representation_type = "MockType"
    c2.source_strategy = "MockStrategy"
    c2.confidence = 0.7
    c2.verification_status = VerificationStatus.CONTRADICTED
    c2.provenance = {'origin': 'test2'}
    c2.verification_results = [{'example_index': 0, 'status': 'Fail'}] # Add some detail

    c3 = MagicMock(spec=CandidateProgram)
    c3.program_representation = "Error Rep"
    c3.representation_type = "MockType"
    c3.source_strategy = "MockStrategy"
    c3.confidence = None
    c3.verification_status = VerificationStatus.ERROR
    c3.provenance = {'origin': 'test3'}
    c3.verification_results = {'error': 'Mock solver timed out'}

    c4 = MagicMock(spec=CandidateProgram)
    c4.program_representation = "Consistent Rep 2"
    c4.representation_type = "MockType"
    c4.source_strategy = "MockStrategy"
    c4.confidence = 0.85
    c4.verification_status = VerificationStatus.CONSISTENT
    c4.provenance = {'origin': 'test4'}
    c4.verification_results = None

    return [c1, c2, c3, c4]

@pytest.fixture
def sample_task_spec_report():
    """Provides a sample TaskSpec for reporting tests."""
    # Simplified for clarity, adjust inputs/outputs as needed
    # Removed invalid 'labels' argument
    return TaskSpec(task_id="report_test_001", inputs=[{'in': 1}], outputs=[{'out': 1}])

@pytest.fixture
def sample_run_summary():
    """Provides a sample run summary dictionary."""
    return {
        'run_timestamp': datetime.datetime.now().isoformat(),
        'total_time_s': 10.5,
        'strategies_run': ['Strat1', 'Strat2'],
        'errors': []
    }

@pytest.fixture
def mock_task_spec_empty():
    """Provides a TaskSpec with no examples."""
    return TaskSpec(task_id="empty_task")

@pytest.fixture
def mock_task_spec_with_examples():
    """Provides a TaskSpec with some simple examples."""
    return TaskSpec(
        task_id="task_with_examples",
        inputs=[{"in": 1}, {"in": 2}],
        outputs=[{"out": "A"}, {"out": "B"}],
        metadata={"domain": "test"}
    )

@pytest.fixture
def mock_task_spec_tictactoe_small():
    """Provides a TaskSpec similar to the small TicTacToe task."""
    # Simplified version for testing reporter structure
    return TaskSpec(
        task_id="tictactoe_standard_small",
        description="TicTacToe task for testing.",
        inputs=[
            {"board": ".........", "player": "X"},
            {"board": "X........", "player": "O"}
        ],
        outputs=[
            {"move": 4}, # Example output
            {"move": 1}  # Example output
        ],
        constraints=["Constraint A"],
        metadata={"label_space": ["move"]}
    )

@pytest.fixture
def reporter_instance(tmp_path):
    """Provides a Reporter instance with a temporary output directory."""
    # Ensure the default dir name is used if not specified
    reporter = Reporter(output_dir=str(tmp_path / "test_reporter_output")) 
    return reporter

# --- Tests ---

def test_reporter_init():
    """Test Reporter initialization."""
    # Test default directory
    reporter = Reporter()
    assert reporter.output_dir == "results"
    # Test custom directory
    reporter = Reporter(output_dir="./custom_reports")
    assert reporter.output_dir == "./custom_reports"

@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs") # Mock makedirs to avoid side effects
@patch("json.dump") # Add patch for json.dump
@patch("framework.reporter.logger") # Mock logger within the module
def test_reporter_generate_report_json(mock_logger, mock_json_dump, mock_makedirs, mock_file, sample_task_spec_report, mock_candidates_list, sample_run_summary):
    """Test JSON report generation."""
    reporter = Reporter(output_dir="./test_report_dir")
    timestamp_str = sample_run_summary['run_timestamp'] # Original timestamp
    safe_timestamp = timestamp_str.replace(':', '-').replace('.', '_') # Sanitized timestamp

    reporter.generate_report(sample_task_spec_report, mock_candidates_list, timestamp_str) # Pass original timestamp

    # Check if output dir was created
    mock_makedirs.assert_called_once_with("./test_report_dir", exist_ok=True)

    # Check if JSON file was opened correctly using the *sanitized* timestamp
    expected_path = os.path.join("./test_report_dir", f"report_{sample_task_spec_report.task_id}_{safe_timestamp}.json")
    # Check if *any* call matches this path, as head report is also written
    mock_file.assert_any_call(expected_path, 'w')

    # Verify the data structure passed to json.dump for the main report
    # Find the call where the first argument is a dict (the report data)
    dump_call_args = None
    for call_args, call_kwargs in mock_json_dump.call_args_list:
         # Check if the first arg is a dict and looks like our report structure
         if isinstance(call_args[0], dict) and 'task_spec' in call_args[0]:
             dump_call_args = call_args[0]
             break

    assert dump_call_args is not None, "json.dump was not called with the expected report data structure"
    report_data = dump_call_args

    # --- Start Verification of JSON Content --- 
    assert report_data["task_spec"]["task_id"] == "report_test_001"
    assert report_data["run_timestamp"] == timestamp_str
    assert report_data["total_candidates_generated"] == len(mock_candidates_list)
    # Check summary counts based on mock_candidates_list structure
    # mock_candidates_list has 2 Consistent, 1 Contradicted, 1 Error
    assert report_data["results_summary"]["CONSISTENT"] == 2
    assert report_data["results_summary"]["CONTRADICTED"] == 1
    assert report_data["results_summary"]["ERROR"] == 1
    assert report_data["results_summary"]["NOT_VERIFIED"] == 0

    assert len(report_data["candidates"]) == len(mock_candidates_list)
    # Check serialization of the first candidate (assuming it's the consistent one)
    # These depend on how _serialize_candidate handles mocks now
    first_serialized_candidate = report_data["candidates"][0]
    assert first_serialized_candidate['source_strategy'] is not None # Check basic fields were serialized
    assert first_serialized_candidate['representation_type'] is not None
    assert first_serialized_candidate['verification_status'] == VerificationStatus.CONSISTENT.name
    assert first_serialized_candidate['confidence'] is not None
    assert isinstance(first_serialized_candidate['provenance'], dict)
    # --- End Verification of JSON Content --- 

def test_reporter_generate_report_logs(reporter_instance, caplog, sample_task_spec_report, sample_run_summary, mock_candidates_list):
    """Test that generate_report logs expected informational messages."""
    reporter = reporter_instance
    caplog.set_level(logging.INFO)
    timestamp_str = sample_run_summary['run_timestamp'] # Get original timestamp
    safe_timestamp = timestamp_str.replace(':', '-').replace('.', '_') # Sanitized timestamp

    # Define expected log messages using the *sanitized* timestamp for filenames
    json_filename = os.path.join(reporter.output_dir, f"report_{sample_task_spec_report.task_id}_{safe_timestamp}.json")
    head_filename = os.path.join(reporter.output_dir, f"report_head_{sample_task_spec_report.task_id}_{safe_timestamp}.json")

    # Pass the actual arguments needed by generate_report (original timestamp)
    reporter.generate_report(sample_task_spec_report, mock_candidates_list, timestamp_str)

    # Extract log messages
    actual_logs = [record.message for record in caplog.records if record.levelno == logging.INFO]

    # Define expected sequence of log messages based on mock_candidates_list
    # mock_candidates_list has 2 Consistent, 1 Contradicted, 1 Error
    expected_info_logs = [
        "Generating reports...",
        f"JSON report saved to: {json_filename}", # Uses safe_timestamp
        f"Head report (first 500 lines) saved to: {head_filename}", # Uses safe_timestamp
        "\n--- Run Summary ---", # Add newline based on implementation
        f"Task ID: {sample_task_spec_report.task_id}",
        f"Timestamp: {timestamp_str}", # Log should show original timestamp
        f"Total Candidates Generated: {len(mock_candidates_list)}", # Count from fixture
        "Verification Summary:",
        f"  - {VerificationStatus.NOT_VERIFIED.name}: 0",
        f"  - {VerificationStatus.CONSISTENT.name}: 2",
        f"  - {VerificationStatus.CONTRADICTED.name}: 1",
        f"  - {VerificationStatus.ERROR.name}: 1",
        "\nTop Consistent Candidates (if any):", # Add newline
        # Add lines for the top 2 consistent candidates (ANY placeholders are okay)
        ANY, # 1st Consistent: Header
        ANY, # 1st Consistent: Representation
        ANY, # 1st Consistent: Provenance
        ANY, # 2nd Consistent: Header
        ANY, # 2nd Consistent: Representation
        ANY, # 2nd Consistent: Provenance
        "--- End Summary ---\n" # Add newline
    ]

    # Check log count first
    assert len(actual_logs) == len(expected_info_logs), \
        f"Expected {len(expected_info_logs)} logs, got {len(actual_logs)}\nActual logs:\n" + "\n".join(actual_logs)

    # Check log content
    for actual, expected in zip(actual_logs, expected_info_logs):
         if expected is ANY:
             continue # Skip comparison for ANY placeholders
         assert actual == expected, f"Log mismatch: Expected '{expected}', got '{actual}'"

@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
@patch("json.dump")
@patch("framework.reporter.logger")
def test_reporter_generate_report_no_candidates(mock_logger, mock_json_dump, mock_makedirs, mock_file, reporter_instance, sample_task_spec_report, sample_run_summary):
    """Test reporting when no candidates are generated."""
    # Use the provided reporter_instance, don't create a new one
    timestamp_str = sample_run_summary['run_timestamp']
    reporter_instance.generate_report(sample_task_spec_report, [], timestamp_str)

    # makedirs is called in the fixture's Reporter init, not here.
    # Check file writes
    assert mock_file.call_count >= 1 # Should write at least the main JSON

    # Check the structure passed to json.dump
    # Get the data from the first argument of the last call to json.dump
    # Note: Using mock_open can be tricky for capturing exact write content.
    # It might be better to assert logger output or check file existence if not mocking open.
    # For now, assume mock_json_dump captured the correct data.
    # Check that json.dump was actually called
    mock_json_dump.assert_called()
    dump_args, _ = mock_json_dump.call_args # Now this should work

    report_data = dump_args[0]

    assert report_data["candidates"] == []
    assert report_data["results_summary"]["NOT_VERIFIED"] == 0
    assert report_data["results_summary"]["CONSISTENT"] == 0
    assert report_data["run_timestamp"] == timestamp_str # Check timestamp string

@pytest.mark.parametrize("num_candidates", [0, 1, 2])
@patch('builtins.open', new_callable=MagicMock)
@patch('os.makedirs')
@patch('json.dump')
@patch('json.dumps')
@patch('framework.reporter.logger')
def test_reporter_generate_report_candidate_counts(mock_logger, mock_json_dumps, mock_json_dump, mock_makedirs, mock_open, reporter_instance, sample_task_spec_report, sample_run_summary, mock_candidate_program, num_candidates):
    """Test that the candidate count in the summary is correct."""
    candidates = [mock_candidate_program] * num_candidates
    timestamp_str = sample_run_summary['run_timestamp']
    reporter_instance.generate_report(sample_task_spec_report, candidates, timestamp_str)

    dump_args, _ = mock_json_dump.call_args
    report_data = dump_args[0]
    # Check the count under the correct key
    assert report_data['results_summary'][VerificationStatus.NOT_VERIFIED.name] == num_candidates
    assert len(report_data['candidates']) == num_candidates
    # Check the timestamp string is stored correctly
    assert report_data['run_timestamp'] == timestamp_str

@patch("builtins.open", new_callable=mock_open)
@patch("os.makedirs")
@patch("json.dump")
@patch("framework.reporter.logger")
def test_reporter_generate_report_no_candidates_json(mock_logger, mock_json_dump, mock_makedirs, mock_file, reporter_instance, sample_task_spec_report, sample_run_summary):
    """Test the JSON structure specifically when no candidates are generated."""
    # Use the fixture instance and pass timestamp string
    timestamp_str = sample_run_summary['run_timestamp']
    reporter_instance.generate_report(sample_task_spec_report, [], timestamp_str)

    # Check data passed to json.dump
    dump_args, _ = mock_json_dump.call_args
    report_data = dump_args[0]

    try:
        assert "task_spec" in report_data
        assert "run_timestamp" in report_data 
        assert report_data["run_timestamp"] == timestamp_str # Check it's the string
        assert "results_summary" in report_data 
        assert "candidates" in report_data
        assert report_data["results_summary"][VerificationStatus.CONSISTENT.name] == 0 
        assert report_data["candidates"] == []
    except KeyError as e:
        pytest.fail(f"Missing key in report JSON: {e}\nReport Data: {report_data}")
    except Exception as e:
        pytest.fail(f"JSON structure check failed or test error: {e}\nReport Data: {report_data}")

def test_reporter_init(reporter_instance):
    """Test Reporter initialization."""
    assert reporter_instance.output_dir.endswith("test_reporter_output")
    assert os.path.isabs(reporter_instance.output_dir)

# Add more tests: e.g., different candidate statuses, edge cases, error handling if dir doesn't exist?
# Test filename generation uniqueness if needed. 