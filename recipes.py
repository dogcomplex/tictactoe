import torch
import time
import random

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Ensure we're using CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Combined Tokens Mapping
token_to_index = {
    'E1': 0,  'E2': 1,  'E3': 2,  'E4': 3,  'E5': 4,  'E6': 5,  'E7': 6,  'E8': 7,  'E9': 8,
    'X1': 9, 'X2': 10, 'X3': 11, 'X4': 12, 'X5': 13, 'X6': 14, 'X7': 15, 'X8': 16, 'X9': 17,
    'O1': 18, 'O2': 19, 'O3': 20, 'O4': 21, 'O5': 22, 'O6': 23, 'O7': 24, 'O8': 25, 'O9': 26,
    'C': 27, 'W': 28, 'L': 29, 'D': 30, 'E': 31,
}

index_to_token = {v: k for k, v in token_to_index.items()}
num_tokens = len(token_to_index)

def generate_valid_hypotheses(batch_size=10000000):
    # 50000000 for 24gb vram reasonable
    """Generate valid hypotheses efficiently using bitwise operations."""
    num_positions = 9
    position_options = 7  # 7 possible states for each position
    output_options = 31   # 31 possible output states
    total_hypotheses = position_options**num_positions * output_options

    print(f"Generating {total_hypotheses:,} hypotheses in batches of {batch_size:,}")
    start_time = time.time()
    batches_generated = 0

    for start in range(0, total_hypotheses, batch_size):
        end = min(start + batch_size, total_hypotheses)
        batch_size = end - start

        # Generate position states
        position_states = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(1)
        position_states = (position_states // (position_options ** torch.arange(num_positions, device=device))) % position_options

        # Generate hypotheses using bitwise operations
        hypotheses = torch.ones((batch_size, num_tokens), dtype=torch.bool, device=device)
        
        # Set position tokens (inverted logic)
        for i in range(num_positions):
            hypotheses[:, i] = ~(position_states[:, i] & 1).bool()
            hypotheses[:, i+9] = ~((position_states[:, i] >> 1) & 1).bool()
            hypotheses[:, i+18] = ~((position_states[:, i] >> 2) & 1).bool()

        # Set output tokens (inverted logic)
        output_states = torch.arange(start, end, device=device) % output_options
        for i in range(5):
            hypotheses[:, 27+i] = ~((output_states >> i) & 1).bool()

        batches_generated += 1
        if batches_generated % 100 == 0:
            progress = (start + batch_size) / total_hypotheses * 100
            elapsed_time = time.time() - start_time
            print(f"Progress: {progress:.2f}% | Batches: {batches_generated:,} | Time: {elapsed_time:.2f}s")

        yield hypotheses

    total_time = time.time() - start_time
    print(f"Total hypotheses generated: {total_hypotheses:,}")
    print(f"Total generation time: {total_time:.2f}s")

def validate_hypotheses(hypothesis_generator, observation_tensors, batch_size=10000000):
    """Validate hypotheses against observations using tensor operations and batching."""
    total_processed = 0
    start_time = time.time()
    
    num_observations = len(observation_tensors)
    obs_inputs = observation_tensors[:, :27]
    obs_outputs = observation_tensors[:, 27:]
    
    for i, hypotheses_batch in enumerate(hypothesis_generator):
        batch_size = hypotheses_batch.shape[0]
        
        # Separate input and output parts of hypotheses
        hyp_inputs = hypotheses_batch[:, :27]
        hyp_outputs = hypotheses_batch[:, 27:]
        
        # Check input matches (hypothesis must have all TRUE flags that observation has)
        input_matches = torch.all(hyp_inputs.unsqueeze(1) | ~obs_inputs.unsqueeze(0), dim=2)
        
        # For matching inputs, check output matches
        # Hypothesis must allow the observation (observation must be a subset of hypothesis)
        output_matches = torch.all(hyp_outputs.unsqueeze(1) | ~obs_outputs.unsqueeze(0), dim=2)
        
        # Calculate counters for this batch
        batch_valid_counts = torch.sum(input_matches & output_matches, dim=1)
        batch_invalid_counts = torch.sum(input_matches & ~output_matches, dim=1)
        batch_miss_counts = num_observations - (batch_valid_counts + batch_invalid_counts)
        
        # Create batch results tensor
        batch_results = torch.stack([batch_miss_counts, batch_valid_counts, batch_invalid_counts], dim=1)
        
        total_processed += batch_size
        
        if (i + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {total_processed:,} hypotheses | Time: {elapsed_time:.2f}s")
        
        yield batch_results

    total_time = time.time() - start_time
    print(f"Validation complete. Total time: {total_time:.2f}s")

def process_observations(observations):
    """Process observations into tensors."""
    return torch.stack([map_observation_to_tensor(board, output_char) for board, output_char in observations])

def map_observation_to_tensor(board, output_char):
    """Map board string and output character to a tensor of token indices."""
    tokens = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    for idx, val in enumerate(board):
        position = str(idx + 1)
        if val == '0':
            token = f'E{position}'
        elif val == '1':
            token = f'X{position}'
        elif val == '2':
            token = f'O{position}'
        token_idx = token_to_index[token]
        tokens[token_idx] = True
    
    output_idx = token_to_index.get(output_char, token_to_index['E'])
    tokens[output_idx] = True
    return tokens

def visualize_hypothesis(hypothesis):
    # Convert hypothesis to a list if it's a tensor
    if torch.is_tensor(hypothesis):
        hypothesis = hypothesis.tolist()
    
    position_tokens = []
    for i in range(9):
        state = (hypothesis[i], hypothesis[i+9], hypothesis[i+18])
        if state == (1, 0, 0):
            position_tokens.append(f"E{i+1}")
        elif state == (0, 1, 0):
            position_tokens.append(f"X{i+1}")
        elif state == (0, 0, 1):
            position_tokens.append(f"O{i+1}")
        elif state == (1, 1, 0):
            position_tokens.append(f"!O{i+1}")
        elif state == (1, 0, 1):
            position_tokens.append(f"!X{i+1}")
        elif state == (0, 1, 1):
            position_tokens.append(f"!E{i+1}")
        elif state == (1, 1, 1):
            position_tokens.append("")
        else:
            position_tokens.append("?")  # Handle any unexpected states

    output_tokens = []
    output_mapping = {27: 'C', 28: 'W', 29: 'L', 30: 'D', 31: 'E'}
    for i in range(27, 32):
        if hypothesis[i]:
            output_tokens.append(output_mapping[i])
    
    position_str = " ".join(token for token in position_tokens if token)
    output_str = " | ".join(output_tokens) if output_tokens else "?"
    
    # filter out double spaces:
    position_str = position_str.replace("  ", " ")

    return f"{position_str} => {output_str}"

def validate_all_hypotheses(results_generator, hypothesis_generator, num_observations, sample_size=20):
    """Process all hypothesis validation results and collect statistics and samples."""
    total_hypotheses = 0
    fully_valid = 0
    partially_valid = 0
    invalid = 0
    
    fully_valid_samples = []
    partially_valid_samples = []
    invalid_samples = []
    
    for batch_results, hypotheses_batch in zip(results_generator, hypothesis_generator):
        total_hypotheses += len(batch_results)
        
        fully_valid_mask = batch_results[:, 1] == num_observations
        partially_valid_mask = (batch_results[:, 1] > 0) & (batch_results[:, 2] == 0)
        invalid_mask = batch_results[:, 2] > 0
        
        fully_valid += torch.sum(fully_valid_mask).item()
        partially_valid += torch.sum(partially_valid_mask).item()
        invalid += torch.sum(invalid_mask).item()
        
        # Sample results and hypotheses for each category
        for mask, samples in [(fully_valid_mask, fully_valid_samples),
                              (partially_valid_mask, partially_valid_samples),
                              (invalid_mask, invalid_samples)]:
            if len(samples) < sample_size:
                indices = torch.where(mask)[0]
                num_to_sample = min(sample_size - len(samples), len(indices))
                sampled_indices = indices[torch.randperm(len(indices))[:num_to_sample]]
                samples.extend([(hypotheses_batch[i].tolist(), batch_results[i].tolist()) for i in sampled_indices])
    
    return {
        'total_hypotheses': total_hypotheses,
        'fully_valid': fully_valid,
        'partially_valid': partially_valid,
        'invalid': invalid,
        'fully_valid_samples': fully_valid_samples,
        'partially_valid_samples': partially_valid_samples,
        'invalid_samples': invalid_samples
    }

def print_shortest_samples(samples, sample_type, limit=20):
    print(f"\nSample of {sample_type} Hypotheses (Shortest 20):")
    sorted_samples = sorted(samples, key=lambda x: len(visualize_hypothesis(x[0])))
    for hypothesis, (misses, valid, invalid) in sorted_samples[:limit]:
        vis = visualize_hypothesis(hypothesis)
        raw_flags = ''.join('1' if flag else '0' for flag in hypothesis)
        print(f"{vis} ::: Misses: {misses}, Valid: {valid}, Invalid: {invalid}")
        print(f"Raw flags: {raw_flags}")
        print("---")

def main(observations):
    torch.cuda.empty_cache()  # Clear CUDA cache before starting
    print("Starting hypothesis generation and validation...")
    start_time = time.time()
    
    batch_size = 20000000  # Adjust this based on your GPU memory
    hypothesis_generator = generate_valid_hypotheses(batch_size)
    observation_tensors = process_observations(observations)
    
    print(f"Observation tensors: {observation_tensors}")
    
    results_generator = validate_hypotheses(hypothesis_generator, observation_tensors, batch_size)
    
    # Create a new hypothesis generator for sampling
    sample_hypothesis_generator = generate_valid_hypotheses(batch_size)
    
    validation_results = validate_all_hypotheses(results_generator, sample_hypothesis_generator, len(observations))
    
    print("\nValidation Results:")
    print(f"Total hypotheses: {validation_results['total_hypotheses']:,}")
    print(f"Fully valid hypotheses: {validation_results['fully_valid']:,}")
    print(f"Partially valid hypotheses: {validation_results['partially_valid']:,}")
    print(f"Invalid hypotheses: {validation_results['invalid']:,}")
    
    print_shortest_samples(validation_results['fully_valid_samples'], "Fully Valid")
    print_shortest_samples(validation_results['partially_valid_samples'], "Partially Valid")
    print_shortest_samples(validation_results['invalid_samples'], "Invalid")
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f}s")

# Example Observations
observations = [
    ('000000000', 'C'),     # Round 1
    ('020020110', 'C'),     # Round 2
    ('110212221', 'W'),     # Round 3
]

# Run the Algorithm
if __name__ == "__main__":
    main(observations)
