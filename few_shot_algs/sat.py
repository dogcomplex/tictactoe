import itertools
from pysat.formula import CNF
from pysat.solvers import Solver
from few_shot_algs.few_shot_alg import Algorithm
import random
import time
import sys

class SATAlgorithm(Algorithm):
    def __init__(self, num_inputs=9, input_bits=3, num_outputs=5):
        super().__init__()
        self.num_inputs = num_inputs
        self.input_bits = input_bits
        self.num_outputs = num_outputs
        self.observations = []
        self.round_count = 0
        self.cache = {}
        self.cnf = CNF()
        self.initialize_general_constraints()

    def initialize_general_constraints(self):
        # Add constraints for each input and output position (exactly one state at a time)
        for i in range(self.num_inputs):
            # At least one state is true
            self.cnf.append([self._var_input_bit(i, b) for b in range(self.input_bits)])
            # At most one state is true
            for b1, b2 in itertools.combinations(range(self.input_bits), 2):
                self.cnf.append([-self._var_input_bit(i, b1), -self._var_input_bit(i, b2)])

        # At least one output state is true
        self.cnf.append([self._var_output_bit(b) for b in range(self.num_outputs)])
        # At most one output state is true
        for b1, b2 in itertools.combinations(range(self.num_outputs), 2):
            self.cnf.append([-self._var_output_bit(b1), -self._var_output_bit(b2)])

    def _var_input_bit(self, i, b):
        return i * self.input_bits + b + 1

    def _var_output_bit(self, b):
        return self.num_inputs * self.input_bits + b + 1

    def _input_to_binary(self, input_str):
        binary = ''
        for digit in input_str:
            if digit == '0':
                binary += '100'
            elif digit == '1':
                binary += '010'
            elif digit == '2':
                binary += '001'
        return binary

    def _output_to_binary(self, output):
        if output == 0:  # 'C'
            return '10000'
        elif output == 1:  # 'W'
            return '01000'
        elif output == 2:  # 'L'
            return '00100'
        elif output == 3:  # 'D'
            return '00010'
        elif output == 4:  # 'E'
            return '00001'

    def _binary_to_output(self, binary):
        return binary.index('1')

    def remove_subsumed_clauses(self, cnf):
        original_clause_count = len(cnf.clauses)
        new_clauses = []
        for i, clause in enumerate(cnf.clauses):
            is_subsumed = False
            for j, other_clause in enumerate(cnf.clauses):
                if i != j and set(clause).issubset(set(other_clause)):
                    is_subsumed = True
                    break
            if not is_subsumed:
                new_clauses.append(clause)
        
        simplified_cnf = CNF(from_clauses=new_clauses)
        removed_clauses = original_clause_count - len(simplified_cnf.clauses)
        
        print(f"Clause simplification: Removed {removed_clauses} redundant clauses")
        print(f"Original clauses: {original_clause_count}, Simplified clauses: {len(simplified_cnf.clauses)}")
        
        return simplified_cnf

    def predict(self, observation: str) -> int:
        # Check if the observation is in the cache
        if observation in self.cache:
            print(f"Cache hit for observation: {observation}")
            return self.cache[observation]

        input_binary = self._input_to_binary(observation)
        
        # Create a new CNF for this prediction
        prediction_cnf = CNF()
        prediction_cnf.extend(self.cnf.clauses)  # Add all existing constraints
        
        # Add constraints for the current input
        for i, bit in enumerate(input_binary):
            if bit == '1':
                prediction_cnf.append([self._var_input_bit(i // self.input_bits, i % self.input_bits)])
            else:
                prediction_cnf.append([-self._var_input_bit(i // self.input_bits, i % self.input_bits)])

        print(f"Number of clauses for prediction: {len(prediction_cnf.clauses)}")
        print(f"Number of variables: {self.num_inputs * self.input_bits + self.num_outputs}")

        with Solver(bootstrap_with=prediction_cnf.clauses) as solver:
            start_time = time.time()
            solutions = []
            
            for model in solver.enum_models():
                solutions.append(model)
                if time.time() - start_time >= 1.0 and solutions:
                    break
            
            if len(solutions) == 0:
                print("Error: No solutions found. This indicates a contradiction in the constraints.")
                print("Current observation:", observation)
                print("Previous observations:", self.observations)
                raise Exception("No solutions found")
            else:
                chosen_model = random.choice(solutions)
                output_bits = ''.join(['1' if self._var_output_bit(b) in chosen_model else '0' for b in range(self.num_outputs)])
                result = self._binary_to_output(output_bits)
                print(f"Multiple solutions found. Output bits: {output_bits}, Result: {result}")
                print(f"Number of solutions found: {len(solutions)}")
                # Store the result in the cache before returning
                self.cache[observation] = result
                return result

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        
        # Update the cache with the correct label
        self.cache[observation] = correct_label

        input_binary = self._input_to_binary(observation)
        self.observations.append((input_binary, correct_label))
        
        # Add constraint for the new observation
        input_vars = [self._var_input_bit(i // self.input_bits, i % self.input_bits) 
                      for i, b in enumerate(input_binary) if b == '1']
        output_vars = [self._var_output_bit(b) 
                       for b, v in enumerate(self._output_to_binary(correct_label)) if v == '1']
        self.cnf.append([-v for v in input_vars] + output_vars)

        print(f"New observation: input={input_binary}, output={correct_label}")
        print(f"Total observations: {len(self.observations)}")
        print(f"Cache size: {len(self.cache)}")
        print(f"Total clauses: {len(self.cnf.clauses)}")
        
        self.round_count += 1

        # Perform clause reduction every 500 rounds
        if self.round_count % 500 == 0:
            print("Performing clause reduction...")
            self.cnf = self.remove_subsumed_clauses(self.cnf)
            print("Clause reduction complete.")
            print(f"Total clauses: {len(self.cnf.clauses)}")
        else:
            print("Skipping clause reduction this round.")

# Usage
algorithm = SATAlgorithm()
