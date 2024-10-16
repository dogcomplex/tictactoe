import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Ensure we're using CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Combined Tokens Mapping
token_to_index = {
    '1': 0,  '2': 1,  '3': 2,  '4': 3,  '5': 4,  '6': 5,  '7': 6,  '8': 7,  '9': 8,
    'X1': 9, 'X2': 10, 'X3': 11, 'X4': 12, 'X5': 13, 'X6': 14, 'X7': 15, 'X8': 16, 'X9': 17,
    'O1': 18, 'O2': 19, 'O3': 20, 'O4': 21, 'O5': 22, 'O6': 23, 'O7': 24, 'O8': 25, 'O9': 26,
    'C': 27, 'W': 28, 'L': 29, 'D': 30, 'E': 31,
}

index_to_token = {v: k for k, v in token_to_index.items()}
num_tokens = len(token_to_index)

# Define the constraints as boolean tensors
POSITION_CONSTRAINTS = torch.tensor([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
], dtype=torch.bool, device=device)

OUTPUT_CONSTRAINTS = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
], dtype=torch.bool, device=device)

XOR_CONSTRAINTS = torch.cat([POSITION_CONSTRAINTS, OUTPUT_CONSTRAINTS], dim=0)
XOR_CONSTRAINTS = XOR_CONSTRAINTS.to(device)


def generate_valid_hypotheses_blind_constrained_search():
    """Generate all valid hypotheses using GPU-intensive operations."""
    # Aim to use about 20GB of VRAM (leaving some headroom)
    batch_size = 50_000_000  # This should use about 4GB of VRAM for 32 tokens
    num_batches = (2**num_tokens + batch_size - 1) // batch_size
    valid_hypotheses = []

    for batch in range(num_batches):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, 2**num_tokens)
        
        # Generate indices for this batch
        indices = torch.arange(start, end, dtype=torch.int64, device=device)
        
        # Generate hypotheses for this batch (uses about 4GB for 1B hypotheses with 32 tokens)
        hypotheses = torch.zeros((len(indices), num_tokens), dtype=torch.bool, device=device)
        for j in range(num_tokens):
            hypotheses[:, j] = (indices & (1 << j)).bool()
        
        # Validate hypotheses against XOR_CONSTRAINTS (temporary float conversion)
        constraint_sums = torch.mm(hypotheses.float(), XOR_CONSTRAINTS.float().t())
        valid_mask = torch.all(constraint_sums == 1, dim=1)
        
        valid_hypotheses.append(hypotheses[valid_mask])
        del hypotheses, constraint_sums  # Explicitly free large tensors
        torch.cuda.empty_cache()  # Clear CUDA cache to free up memory
        
        print(f"Processed batch {batch + 1}/{num_batches}, Total valid: {sum(h.shape[0] for h in valid_hypotheses)}")

    return torch.cat(valid_hypotheses, dim=0)


def generate_valid_hypotheses():
    """Generate all valid hypotheses efficiently using tensor operations, knowing the constraints."""
    num_positions = 9
    position_options = 3  # 'empty', 'X', 'O'
    output_options = 5    # 'C', 'W', 'L', 'D', 'E'

    # Total combinations for positions and outputs
    total_position_combinations = position_options ** num_positions  # 19,683
    total_hypotheses = total_position_combinations * output_options  # 98,415

    # Generate position indices (0 to 19,682)
    positions_indices = torch.arange(total_position_combinations, device=device)

    # Convert to base-3 digits to get options per position
    positions_digits = positions_indices.unsqueeze(1) // (3 ** torch.arange(num_positions-1, -1, -1, device=device)) % 3
    # positions_digits shape: (19,683, 9), values: 0, 1, 2

    # Map digits to token indices per position
    mapping_digits_to_tokens = torch.zeros(num_positions, position_options, dtype=torch.long, device=device)
    for i in range(num_positions):
        mapping_digits_to_tokens[i, 0] = token_to_index[str(i+1)]       # Empty
        mapping_digits_to_tokens[i, 1] = token_to_index[f'X{i+1}']     # 'X' in position i+1
        mapping_digits_to_tokens[i, 2] = token_to_index[f'O{i+1}']     # 'O' in position i+1

    # Get token indices per hypothesis for positions
    positions_token_indices = mapping_digits_to_tokens[
        torch.arange(num_positions, device=device).unsqueeze(0),
        positions_digits
    ]  # Shape: (19,683, 9)

    # Output token indices
    output_token_indices = torch.tensor(
        [token_to_index[token] for token in ['C', 'W', 'L', 'D', 'E']],
        device=device
    )  # Shape: (5,)

    # Expand positions and outputs to get all combinations
    positions_token_indices_expanded = positions_token_indices.repeat_interleave(output_options, dim=0)
    output_token_indices_expanded = output_token_indices.repeat(total_position_combinations)

    # Combine token indices
    all_token_indices = torch.cat(
        [positions_token_indices_expanded, output_token_indices_expanded.unsqueeze(1)],
        dim=1
    )  # Shape: (98,415, 10)

    # Create hypotheses tensor
    hypotheses = torch.zeros((total_hypotheses, num_tokens), dtype=torch.bool, device=device)
    hypotheses[torch.arange(total_hypotheses, device=device).unsqueeze(1), all_token_indices] = True

    return hypotheses

def process_observations(observations):
    """Process observations into tensors."""
    return torch.stack([map_observation_to_tensor(board, output_char) for board, output_char in observations])

def validate_hypotheses(hypotheses, observation_tensors):
    """Validate hypotheses against observations and constraints using tensor operations."""
    num_hypotheses = hypotheses.size(0)
    num_observations = observation_tensors.size(0)
    
    # Expand tensors to align hypotheses and observations
    expanded_observations = observation_tensors.unsqueeze(0).expand(num_hypotheses, -1, -1)
    expanded_hypotheses = hypotheses.unsqueeze(1).expand(-1, num_observations, -1)
    
    # Check if hypothesis conditions are met in observations
    matches = (expanded_hypotheses & expanded_observations).all(dim=2)
    
    # A hypothesis is consistent if it matches all observations
    consistency_mask = matches.all(dim=1)
    
    # Validate hypotheses against constraints
    valid_hypotheses = validate_constraints_tensor(hypotheses)
    
    # Combine consistency and constraint validation
    final_mask = consistency_mask & valid_hypotheses
    
    return hypotheses[final_mask]

def validate_constraints_tensor(tensor):
    """Validate that a tensor meets the defined constraints using tensor operations."""
    constraint_sums = torch.mm(tensor.float(), XOR_CONSTRAINTS.float().t())
    valid_constraints = torch.all(constraint_sums == 1, dim=1)
    return valid_constraints

def main(observations):
    torch.cuda.empty_cache()  # Clear CUDA cache before starting
    hypotheses = generate_valid_hypotheses()
    observation_tensors = process_observations(observations)
    
    # Validate constraints for observations
    valid_observations = validate_constraints_tensor(observation_tensors)
    if not valid_observations.all():
        print("Warning: Some observations do not meet the constraints.")
        observation_tensors = observation_tensors[valid_observations]
    
    consistent_hypotheses = validate_hypotheses(hypotheses, observation_tensors)
    
    print(f"Total valid hypotheses: {hypotheses.shape[0]}")
    print(f"Consistent hypotheses: {consistent_hypotheses.shape[0]}")
    print("\nSample of Consistent Hypotheses:")
    for hypothesis in consistent_hypotheses[:10]:  # Print first 10 for brevity
        tokens = [index_to_token[idx] for idx, val in enumerate(hypothesis) if val]
        print(" ".join(tokens))

def map_observation_to_tensor(board, output_char):
    """Map board string and output character to a tensor of token indices."""
    tokens = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    for idx, val in enumerate(board):
        position = str(idx + 1)
        if val == '0':
            token = position
        elif val == '1':
            token = f'X{position}'
        elif val == '2':
            token = f'O{position}'
        token_idx = token_to_index[token]
        tokens[token_idx] = True
    
    output_idx = token_to_index.get(output_char, token_to_index['E'])
    tokens[output_idx] = True
    return tokens

# Example Observations
observations = [
    ('000000000', 'C'),     # Round 1
    ('020020110', 'C'),     # Round 2
    ('110212221', 'W'),     # Round 3
]

# Run the Algorithm
if __name__ == "__main__":
    main(observations)