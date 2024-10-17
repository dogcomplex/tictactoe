import torch
import time
import random
from typing import List, Tuple
import numpy as np
import os

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

def generate_valid_hypotheses(batch_size):
    """Generate valid hypotheses efficiently using bitwise operations."""
    num_positions = 9
    position_options = 7  # 7 possible states for each position
    output_options = 5    # 5 possible output states (C, W, L, D, E)
    total_hypotheses = position_options**num_positions * output_options

    print(f"Generating {total_hypotheses:,} hypotheses in batches of {batch_size:,}")
    start_time = time.time()
    batches_generated = 0

    for start in range(0, total_hypotheses, batch_size):
        end = min(start + batch_size, total_hypotheses)
        current_batch_size = end - start

        # Generate position states
        position_states = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(1)
        position_states = (position_states // (position_options ** torch.arange(num_positions, device=device))) % position_options

        # Generate hypotheses using bitwise operations
        hypotheses = torch.ones((current_batch_size, 32), dtype=torch.bool, device=device)
        
        # Set position tokens (inverted logic)
        for i in range(num_positions):
            hypotheses[:, i] = ~(position_states[:, i] & 1).bool()
            hypotheses[:, i+9] = ~((position_states[:, i] >> 1) & 1).bool()
            hypotheses[:, i+18] = ~((position_states[:, i] >> 2) & 1).bool()

        # Set output tokens (only one output token is True for each hypothesis)
        output_states = torch.arange(start, end, device=device) % output_options
        hypotheses[:, 27:] = False  # Reset all output tokens to False
        hypotheses[torch.arange(current_batch_size), output_states + 27] = True

        batches_generated += 1
        if batches_generated % 100 == 0:
            progress = (start + current_batch_size) / total_hypotheses * 100
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
    
    if output_char is not None:
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
    valid = 0
    invalid = 0
    
    valid_samples = []
    invalid_samples = []
    
    for batch_results, hypotheses_batch in zip(results_generator, hypothesis_generator):
        total_hypotheses += len(batch_results)
        
        valid_mask = batch_results[:, 2] == 0  # Hypotheses with no invalid matches are considered valid
        invalid_mask = ~valid_mask
        
        valid += torch.sum(valid_mask).item()
        invalid += torch.sum(invalid_mask).item()
        
        # Sample results and hypotheses for each category
        for mask, samples in [(valid_mask, valid_samples),
                              (invalid_mask, invalid_samples)]:
            if len(samples) < sample_size:
                indices = torch.where(mask)[0]
                num_to_sample = min(sample_size - len(samples), len(indices))
                sampled_indices = indices[torch.randperm(len(indices))[:num_to_sample]]
                samples.extend([(hypotheses_batch[i].tolist(), batch_results[i].tolist()) for i in sampled_indices])
    
    return {
        'total_hypotheses': total_hypotheses,
        'valid': valid,
        'invalid': invalid,
        'valid_samples': valid_samples,
        'invalid_samples': invalid_samples
    }

def print_sorted_samples(samples, sample_type, limit=20):
    print(f"\nSample of {sample_type} Hypotheses (Sorted by valid hits, then by conciseness):")
    sorted_samples = sorted(samples, key=lambda x: (-x[1][1], len(visualize_hypothesis(x[0]))))
    for hypothesis, (misses, valid, invalid) in sorted_samples[:limit]:
        vis = visualize_hypothesis(hypothesis)
        raw_flags = ''.join('1' if flag else '0' for flag in hypothesis)
        print(f"{vis} ::: Misses: {misses}, Valid: {valid}, Invalid: {invalid}")
        print(f"Raw flags: {raw_flags}")
        print("---")

def pack_bits(x: torch.Tensor) -> torch.Tensor:
    """Pack boolean tensor into int32 tensor."""
    packed = torch.zeros((x.shape[0],), dtype=torch.int32, device=x.device)
    for i in range(32):
        packed |= x[:, i].to(torch.int32) << i
    return packed

def unpack_bits(x: torch.Tensor) -> torch.Tensor:
    """Unpack int32 tensor into boolean tensor."""
    unpacked = torch.zeros((x.shape[0], 32), dtype=torch.bool, device=x.device)
    for i in range(32):
        unpacked[:, i] = (x & (1 << i)) != 0
    return unpacked

class HypothesisManager:
    def __init__(self, device, use_disk_cache=False):
        self.device = device
        self.use_disk_cache = use_disk_cache
        self.batch_size = 10000000  # Adjust based on your GPU memory
        self.total_hypotheses = 7**9 * 5  # Correct calculation: 7^9 * 5 = 201,326,595
        self.hypotheses_file = 'all_hypotheses.pt'
        self.hypotheses = None
        self.valid_hypotheses = None
        self.initialize()

    def initialize(self):
        print("Initializing hypothesis manager...")
        if self.use_disk_cache and os.path.exists(self.hypotheses_file):
            self.load_hypotheses_from_disk()
        else:
            self.generate_hypotheses()
        self.valid_hypotheses = torch.ones(self.total_hypotheses, dtype=torch.bool, device=self.device)
        print(f"Initialization complete. Total hypotheses: {self.total_hypotheses}")

    def generate_hypotheses(self):
        print("Generating all hypotheses...")
        self.hypotheses = torch.zeros((self.total_hypotheses,), dtype=torch.int32, device=self.device)
        
        start_time = time.time()
        hypotheses_generated = 0
        for batch in generate_valid_hypotheses(self.batch_size):
            end = min(hypotheses_generated + len(batch), self.total_hypotheses)
            packed_batch = pack_bits(batch)
            self.hypotheses[hypotheses_generated:end] = packed_batch[:end-hypotheses_generated]
            hypotheses_generated += len(batch)
            
            if hypotheses_generated % (self.batch_size * 10) == 0:
                elapsed_time = time.time() - start_time
                progress = hypotheses_generated / self.total_hypotheses
                estimated_total_time = elapsed_time / progress
                remaining_time = estimated_total_time - elapsed_time
                print(f"Progress: {progress*100:.2f}% | Estimated time remaining: {remaining_time/60:.2f} minutes")

            if hypotheses_generated >= self.total_hypotheses:
                break

        print(f"All hypotheses generated and stored in VRAM. Total: {hypotheses_generated:,}")
        
        if self.use_disk_cache:
            self.save_hypotheses_to_disk()

    def save_hypotheses_to_disk(self):
        print(f"Saving hypotheses to {self.hypotheses_file}...")
        torch.save(self.hypotheses, self.hypotheses_file)
        print(f"Hypotheses saved to disk. File size: {os.path.getsize(self.hypotheses_file) / (1024**3):.2f} GB")

    def load_hypotheses_from_disk(self):
        print(f"Loading hypotheses from {self.hypotheses_file}...")
        self.hypotheses = torch.load(self.hypotheses_file, map_location=self.device)
        print("Hypotheses loaded into VRAM")

    def get_matching_hypotheses(self, observation: torch.Tensor) -> torch.Tensor:
        matching_indices = []
        obs_mask = (1 << torch.arange(27, device=self.device)).masked_fill(observation[:27] == 0, 0).sum().item()
        
        for batch_start in range(0, self.total_hypotheses, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.total_hypotheses)
            
            batch = self.hypotheses[batch_start:batch_end]
            
            # Check if observation is a bitwise subset of each hypothesis
            input_matches = (batch & obs_mask) == obs_mask
            
            matching_batch_indices = torch.where(input_matches)[0] + batch_start
            matching_indices.append(matching_batch_indices)
        
        return torch.cat(matching_indices) if matching_indices else torch.tensor([], device=self.device)

    def _print_sample_hypotheses(self, sampled_indices, correct_label):
        for idx in sampled_indices:
            hyp = unpack_bits(self.hypotheses[idx].unsqueeze(0)).squeeze(0)
            vis = self.visualize_hypothesis(hyp)
            print(f"{vis} ::: Correct Label: {correct_label}")

    def visualize_hypothesis(self, hypothesis):
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

    @torch.no_grad()
    def update_valid_hypotheses(self, observation: str, correct_label: str):
        # Convert tensors to CPU at the start to reduce GPU memory usage
        obs_tensor = map_observation_to_tensor(observation, correct_label).cpu()
        obs_mask = (1 << torch.arange(27)).masked_fill(obs_tensor[:27] == 0, 0).sum().item()
        correct_output_mask = 1 << token_to_index[correct_label]
        
        total_input_matches = 0
        total_newly_invalid = 0
        prev_valid_count = int(self.valid_hypotheses.sum().item())
        
        newly_still_valid_samples = []
        newly_invalid_samples = []
        sample_size = 10

        for batch_start in range(0, self.total_hypotheses, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.total_hypotheses)
            
            valid_mask = self.valid_hypotheses[batch_start:batch_end].clone().cpu()
            batch = self.hypotheses[batch_start:batch_end][valid_mask].cpu()
            
            if len(batch) > 0:
                input_matches = (batch & obs_mask) == obs_mask
                output_matches = (batch & correct_output_mask) != 0
                
                newly_invalid = input_matches & ~output_matches
                newly_still_valid = input_matches & output_matches
                
                self.valid_hypotheses[batch_start:batch_end][valid_mask] &= ~newly_invalid.to(self.device)
                
                total_input_matches += int(input_matches.sum().item())
                total_newly_invalid += int(newly_invalid.sum().item())

                # Sample newly still valid hypotheses
                if len(newly_still_valid_samples) < sample_size:
                    still_valid_indices = torch.where(newly_still_valid)[0]
                    num_to_sample = min(sample_size - len(newly_still_valid_samples), len(still_valid_indices))
                    if num_to_sample > 0:
                        sampled_indices = still_valid_indices[torch.randperm(len(still_valid_indices))[:num_to_sample]]
                        newly_still_valid_samples.extend(batch[sampled_indices].tolist())

                # Sample newly invalid hypotheses
                if len(newly_invalid_samples) < sample_size:
                    invalid_indices = torch.where(newly_invalid)[0]
                    num_to_sample = min(sample_size - len(newly_invalid_samples), len(invalid_indices))
                    if num_to_sample > 0:
                        sampled_indices = invalid_indices[torch.randperm(len(invalid_indices))[:num_to_sample]]
                        newly_invalid_samples.extend(batch[sampled_indices].tolist())

            # Clear unnecessary tensors
            del valid_mask, batch, input_matches, output_matches, newly_invalid, newly_still_valid
            
        new_valid_count = int(self.valid_hypotheses.sum().item())
        
        unmatched_input_count = prev_valid_count - total_input_matches
        matched_still_valid = total_input_matches - total_newly_invalid
        
        print(f"Update results for observation '{observation}' with label '{correct_label}':")
        print(f"  Previously valid hypotheses: {prev_valid_count:,}")
        print(f"  > Unmatched input hypotheses: {unmatched_input_count:,} ({unmatched_input_count/prev_valid_count:.2%})")
        print(f"  > Input matches: {total_input_matches:,} ({total_input_matches/prev_valid_count:.2%})")
        print(f"      Matched Newly invalid: {total_newly_invalid:,} ({total_newly_invalid/prev_valid_count:.2%})")
        print(f"      - Matched still valid: {matched_still_valid:,} ({matched_still_valid/prev_valid_count:.2%})")
        print(f"  = Valid hypotheses remaining: {new_valid_count:,} ({new_valid_count/self.total_hypotheses:.2%})")
        
        # Print sample hypotheses
        print("\nSample of Valid Hypotheses:")
        for hyp in newly_still_valid_samples:
            vis = self.visualize_hypothesis(unpack_bits(torch.tensor([hyp], device='cpu')).squeeze(0))
            print(f"{vis} ::: Correct Label: {correct_label}")

        print("\nSample of Invalid Hypotheses:")
        for hyp in newly_invalid_samples:
            vis = self.visualize_hypothesis(unpack_bits(torch.tensor([hyp], device='cpu')).squeeze(0))
            print(f"{vis} ::: Correct Label: {correct_label}")

        return {
            'new_valid_count': new_valid_count,
            'unmatched_input_count': unmatched_input_count,
            'matched_still_valid': matched_still_valid,
            'matched_newly_invalid': total_newly_invalid,
            'input_matches': total_input_matches,
        }



    def get_random_valid_hypothesis(self) -> torch.Tensor:
        valid_indices = torch.where(self.valid_hypotheses)[0]
        if len(valid_indices) == 0:
            raise ValueError("No valid hypotheses available")
        random_index = valid_indices[torch.randint(0, len(valid_indices), (1,))]
        hypothesis = unpack_bits(self.hypotheses[random_index].unsqueeze(0))
        return hypothesis.squeeze(0)

    def get_top_hypotheses(self, n=10, observation_tensors=None, matching_indices=None, include_invalid=False, include_miss=False):
        if matching_indices is None:
            indices = torch.arange(self.total_hypotheses, device=self.device)
        else:
            indices = matching_indices
        
        if not include_invalid and not include_miss:
            indices = indices[self.valid_hypotheses[indices]]
        
        if len(indices) == 0:
            return []
        
        sample_size = min(n * 100, len(indices))
        sampled_indices = indices[torch.randperm(len(indices))[:sample_size]]
        
        hypotheses = unpack_bits(self.hypotheses[sampled_indices])
        
        if observation_tensors is not None:
            stats = self.calculate_hypothesis_stats(hypotheses, observation_tensors)
            sorted_hypotheses = sorted(zip(hypotheses, sampled_indices, stats), 
                                       key=lambda x: (-x[2][1], len(visualize_hypothesis(x[0])), x[2][0]))
        else:
            sorted_hypotheses = sorted(zip(hypotheses, sampled_indices), 
                                       key=lambda x: (len(visualize_hypothesis(x[0])), sum(x[0])))
        
        return sorted_hypotheses[:n]

    def calculate_hypothesis_stats(self, hypotheses, observation_tensors):
        obs_inputs = observation_tensors[:, :27]
        obs_outputs = observation_tensors[:, 27:]
        
        input_matches = torch.all(hypotheses[:, :27].unsqueeze(1) | ~obs_inputs.unsqueeze(0), dim=2)
        output_matches = torch.all(hypotheses[:, 27:].unsqueeze(1) | ~obs_outputs.unsqueeze(0), dim=2)
        
        valid_counts = torch.sum(input_matches & output_matches, dim=1)
        invalid_counts = torch.sum(input_matches & ~output_matches, dim=1)
        miss_counts = len(observation_tensors) - (valid_counts + invalid_counts)
        
        return list(zip(miss_counts.tolist(), valid_counts.tolist(), invalid_counts.tolist()))

class TicTacToeAlgorithm:
    def __init__(self, use_disk_cache=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hypothesis_manager = HypothesisManager(self.device, use_disk_cache)
        self.observations = []  # Store the observation history
        self.current_matching_indices = None  # Store matching indices for current observation

    def predict(self, observation: str) -> str:
        obs_tensor = map_observation_to_tensor(observation, None)
        print(f"Observation tensor shape: {obs_tensor.shape}")
        self.current_matching_indices = self.hypothesis_manager.get_matching_hypotheses(obs_tensor)

        if len(self.current_matching_indices) == 0:
            print("No matching hypotheses found. Returning random guess.")
            return random.choice(['C', 'W', 'L', 'D', 'E'])

        # Get all matching hypotheses
        matching_hypotheses = unpack_bits(self.hypothesis_manager.hypotheses[self.current_matching_indices])
        
        # Count occurrences of each output flag
        output_counts = matching_hypotheses[:, 27:].sum(dim=0)
        total_matches = len(self.current_matching_indices)

        # Calculate unweighted probability distribution
        unweighted_probs = output_counts.float() / total_matches

        # Calculate weighted probability distribution
        if len(self.observations) > 0:
            observation_tensors = process_observations(self.observations)
            stats = self.hypothesis_manager.calculate_hypothesis_stats(matching_hypotheses, observation_tensors)
            weights = torch.tensor([stat[1] for stat in stats], device=self.device).float()  # Use valid counts as weights
            weighted_output_counts = (matching_hypotheses[:, 27:].float() * weights.unsqueeze(1)).sum(dim=0)
            weighted_probs = weighted_output_counts / weights.sum()
        else:
            weighted_probs = unweighted_probs

        # Print probability distributions with counts
        print(f"\nProbability distribution of possible responses (Total matches: {total_matches:,}):")
        print("Output | Unweighted Prob | Weighted Prob | Count")
        print("-----------------------------------------------------")
        for i in range(5):
            output_char = index_to_token[i + 27]
            unweighted_prob = unweighted_probs[i].item()
            weighted_prob = weighted_probs[i].item()
            count = output_counts[i].item()
            print(f"{output_char:6} | {unweighted_prob:.6f} | {weighted_prob:.6f} | {count:,}")

        # Choose the output with the highest weighted probability
        most_likely_output = weighted_probs.argmax().item()
        result = index_to_token[most_likely_output + 27]

        # Clear caches and collect garbage
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        return result

    def update_history(self, observation: str, guess, correct_label: str):
        self.observations.append(observation)
        self.guesses.append(guess)
        self.correct_labels.append(correct_label)

        self.hypothesis_manager.update_valid_hypotheses(observation, correct_label)

        # Clear caches and collect garbage
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Reset current_matching_indices for the next prediction
        self.current_matching_indices = None

    def update_history(self, observation: str, guess, correct_label: str):
        print(f"Updating history...")
        if isinstance(guess, int):
            guess = ['C', 'W', 'L', 'D', 'E'][guess]
        elif not isinstance(guess, str):
            print(f"Warning: Unexpected guess type: {type(guess)}. Defaulting to 'E'")
            guess = 'E'
        
        latest_update = self.hypothesis_manager.update_valid_hypotheses(observation, correct_label)

        # Add the new observation to the history
        self.observations.append((observation, correct_label))

        # Clear caches and collect garbage
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Reset current_matching_indices for the next prediction
        self.current_matching_indices = None

    def _print_sample_hypotheses(self, sample_indices, obs_tensor):
        for idx in sample_indices:
            hyp = self.hypothesis_manager.hypotheses[idx]
            vis = visualize_hypothesis(unpack_bits(hyp.unsqueeze(0)).squeeze(0))
            
            # Check if obs_tensor is a tensor or a single value
            if isinstance(obs_tensor, torch.Tensor) and obs_tensor.dim() > 0:
                correct_label = obs_tensor[-1].item()
            else:
                correct_label = obs_tensor  # Assume it's already the correct label
            
            is_valid = (hyp & (1 << token_to_index[correct_label])) != 0
            status = "Valid" if is_valid else "Invalid"
            print(f"{vis} ::: {status}")

    def get_all_observations(self):
        return self.observations

def main(observations: List[Tuple[str, str]]):
    torch.cuda.empty_cache()
    print("Starting hypothesis generation and validation...")
    start_time = time.time()

    algorithm = TicTacToeAlgorithm()

    for i, (board, correct_output) in enumerate(observations):
        guess = algorithm.predict(board)
        print(f"\nRound {i+1}:")
        print(f"Observation: {board}, Guess: {guess}, Correct: {correct_output}")
        algorithm.update_history(board, guess, correct_output)

    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f}s")
    print(f"Valid Hypotheses Remaining: {torch.sum(algorithm.hypothesis_manager.valid_hypotheses).item():,}")

# Example Observations
observations = [
    ('000000000', 'C'),
    ('020020110', 'C'),
    ('110212221', 'W'),
    ('000000000', 'C'),
    ('000000001', 'C'),
]

if __name__ == "__main__":
    main(observations)