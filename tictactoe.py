square_state_space = ['0', '1', '2']
label_space = ['ok', 'win1', 'win2', 'draw', 'error']

def solver(input):
    board = [[input[3*i + j] for j in range(3)] for i in range(3)]
    if '0' in input:
        return 'ok'
    for player in ['1', '2']:
        # Check rows
        for i in range(3):
            if all(board[i][j] == player for j in range(3)):
                return 'win' + player
        # Check columns
        for j in range(3):
            if all(board[i][j] == player for i in range(3)):
                return 'win' + player
        # Check diagonals
        if all(board[i][i] == player for i in range(3)):
            return 'win' + player
        if all(board[i][2 - i] == player for i in range(3)):
            return 'win' + player
    return 'draw'



def function2(input):
    # Define the winning patterns
    winning_patterns = [
        [0, 1, 2],  # Row 1
        [3, 4, 5],  # Row 2
        [6, 7, 8],  # Row 3
        [0, 3, 6],  # Column 1
        [1, 4, 7],  # Column 2
        [2, 5, 8],  # Column 3
        [0, 4, 8],  # Diagonal 1
        [2, 4, 6]   # Diagonal 2
    ]
    
    # Check for Player 1 win
    for pattern in winning_patterns:
        if input[pattern[0]] == input[pattern[1]] == input[pattern[2]] == '1':
            return "win1"
    
    # Check for Player 2 win
    for pattern in winning_patterns:
        if input[pattern[0]] == input[pattern[1]] == input[pattern[2]] == '2':
            return "win2"
    
    # Check if the game is still ongoing (if there are empty spaces)
    if '0' in input:
        return "ok"
    
    # If no winner and no empty spaces, it's a draw
    return "draw"
  
def function3(input):
    # Define the winning patterns
    winning_patterns = [
        [0, 1, 2],  # Row 1
        [3, 4, 5],  # Row 2
        [6, 7, 8],  # Row 3
        [0, 3, 6],  # Column 1
        [1, 4, 7],  # Column 2
        [2, 5, 8],  # Column 3
        [0, 4, 8],  # Diagonal 1
        [2, 4, 6]   # Diagonal 2
    ]
    
    # Check for Player 1 win
    for pattern in winning_patterns:
        if input[pattern[0]] == input[pattern[1]] == input[pattern[2]] == '1':
            return "win1"
    
    # Check for Player 2 win
    for pattern in winning_patterns:
        if input[pattern[0]] == input[pattern[1]] == input[pattern[2]] == '2':
            return "win2"
    
    # Special edge case: if board is full but no winner, it's a draw
    if '0' not in input:
        return "draw"
    
    # If the game is still ongoing (empty spaces exist)
    return "ok"

def function4(input):
    # Define the winning patterns
    winning_patterns = [
        [0, 1, 2],  # Row 1
        [3, 4, 5],  # Row 2
        [6, 7, 8],  # Row 3
        [0, 3, 6],  # Column 1
        [1, 4, 7],  # Column 2
        [2, 5, 8],  # Column 3
        [0, 4, 8],  # Diagonal 1
        [2, 4, 6]   # Diagonal 2
    ]
    
    # Check for Player 1 win
    for pattern in winning_patterns:
        if input[pattern[0]] == input[pattern[1]] == input[pattern[2]] == '1':
            return "win1"
    
    # Check for Player 2 win
    for pattern in winning_patterns:
        if input[pattern[0]] == input[pattern[1]] == input[pattern[2]] == '2':
            return "win2"
    
    # If the game is ongoing (empty spaces exist)
    if '0' in input:
        return "ok"
    
    # If no winner and no empty spaces, it's a draw
    return "draw"


def function5(input):
    # Count the number of moves by each player
    num_1 = input.count('1')
    num_2 = input.count('2')
    
    # Player 1 always starts first; number of '1's should be equal to or one more than number of '2's
    if not (num_1 == num_2 or num_1 == num_2 + 1):
        return "ok"  # Invalid game state
    
    # Define the winning patterns
    winning_patterns = [
        [0, 1, 2],  # Row 1
        [3, 4, 5],  # Row 2
        [6, 7, 8],  # Row 3
        [0, 3, 6],  # Column 1
        [1, 4, 7],  # Column 2
        [2, 5, 8],  # Column 3
        [0, 4, 8],  # Diagonal 1
        [2, 4, 6]   # Diagonal 2
    ]
    
    # Check for wins
    win1 = False
    win2 = False
    for pattern in winning_patterns:
        line = [input[i] for i in pattern]
        if line == ['1', '1', '1']:
            win1 = True
        if line == ['2', '2', '2']:
            win2 = True
    
    # If both players have winning lines, it's an invalid game state
    if win1 and win2:
        return "ok"  # Invalid game state
    
    # If Player 1 wins, number of '1's should be one more than number of '2's
    if win1:
        if num_1 == num_2 + 1:
            return "win1"
        else:
            return "ok"  # Invalid game state
    
    # If Player 2 wins, number of '1's should be equal to number of '2's
    if win2:
        if num_1 == num_2:
            return "win2"
        else:
            return "ok"  # Invalid game state
    
    # If the board is full and no winner, it's a draw
    if '0' not in input:
        return "draw"
    
    # Game is still in progress
    return "ok"




def tictactoe(input):
    # Define win conditions (rows, columns, and diagonals)
    win_conditions = [
        [0, 1, 2],  # Row 1
        [3, 4, 5],  # Row 2
        [6, 7, 8],  # Row 3
        [0, 3, 6],  # Column 1
        [1, 4, 7],  # Column 2
        [2, 5, 8],  # Column 3
        [0, 4, 8],  # Diagonal 1
        [2, 4, 6],  # Diagonal 2
    ]
    
    # Check for Player 1 win
    for condition in win_conditions:
        if input[condition[0]] == input[condition[1]] == input[condition[2]] == '1':
            return "win1"
    
    # Check for Player 2 win
    for condition in win_conditions:
        if input[condition[0]] == input[condition[1]] == input[condition[2]] == '2':
            return "win2"
    
    # Check for draw (no empty spaces and no winner)
    if '0' not in input:
        return "draw"
      
    # count number of 1s and 2s
    count_1 = input.count('1')
    count_2 = input.count('2')
    if count_1 == count_2:
      return "ok"
    if count_1 != count_2 and count_1 != count_2 + 1:
      return "error"
    
    # If no win and there are empty spaces, the game is still in progress
    return "ok"



def tictactoe_no_diags(input):
    # Define win conditions (rows, columns, and diagonals)
    win_conditions = [
        [0, 1, 2],  # Row 1
        [3, 4, 5],  # Row 2
        [6, 7, 8],  # Row 3
        [0, 3, 6],  # Column 1
        [1, 4, 7],  # Column 2
        [2, 5, 8],  # Column 3
        #[0, 4, 8],  # Diagonal 1
        #[2, 4, 6],  # Diagonal 2
    ]
    
    # Check for Player 1 win
    for condition in win_conditions:
        if input[condition[0]] == input[condition[1]] == input[condition[2]] == '1':
            return "win1"
    
    # Check for Player 2 win
    for condition in win_conditions:
        if input[condition[0]] == input[condition[1]] == input[condition[2]] == '2':
            return "win2"
    
    # Check for draw (no empty spaces and no winner)
    if '0' not in input:
        return "draw"
      
    # count number of 1s and 2s
    count_1 = input.count('1')
    count_2 = input.count('2')
    if count_1 == count_2:
      return "ok"
    if count_1 != count_2 and count_1 != count_2 + 1:
      return "error"
    
    # If no win and there are empty spaces, the game is still in progress
    return "ok"


oracle = tictactoe

boards = {}
states = {}
final_states = {}

def move(state):
  if state in states:
    return
  board, moves = state.split('_')
  if oracle(board) == 'error':
    print("error", state)
    return
  states[state] = oracle(board)
  boards[board] = oracle(board)
  if oracle(board) != 'ok': # if tie or win
    final_states[state] = oracle(board)
    return
  count = board.count('0')
  turn = '1' if count % 2 == 1 else '2'
  for i in range(len(board)):
    if board[i] == '0':
      new_board = board[:i] + turn + board[i+1:]
      new_moves = moves + str(i)
      new_state = new_board + '_' + new_moves
      move(new_state)

def init():
  move('000000000_')
  return states

print("Initializing TicTacToe all states...")
states = init()
print("TicTacToe all states initialized.")

import random

def X_turn(input):
  return input.count('1') == input.count('2')








# Check if Player 1 has won
def test_win1(input):
    wins = [
        [0, 1, 2],  # Row 1
        [3, 4, 5],  # Row 2
        [6, 7, 8],  # Row 3
        [0, 3, 6],  # Column 1
        [1, 4, 7],  # Column 2
        [2, 5, 8],  # Column 3
        [0, 4, 8],  # Diagonal 1
        [2, 4, 6],  # Diagonal 2
    ]
    return any(all(input[i] == '1' for i in win) for win in wins)

# Check if Player 2 has won
def test_win2(input):
    wins = [
        [0, 1, 2],  # Row 1
        [3, 4, 5],  # Row 2
        [6, 7, 8],  # Row 3
        [0, 3, 6],  # Column 1
        [1, 4, 7],  # Column 2
        [2, 5, 8],  # Column 3
        [0, 4, 8],  # Diagonal 1
        [2, 4, 6],  # Diagonal 2
    ]
    return any(all(input[i] == '2' for i in win) for win in wins)

# Check if the game is a draw
def test_draw(input):
    # The game is a draw if no empty spaces ('0') exist and neither player has won
    return '0' not in input and not test_win1(input) and not test_win2(input)

# Check if the game is still in progress
def test_in_progress(input):
    # Game is still in progress if there are empty spaces ('0') 
    # and no winner has been determined yet
    return '0' in input and not test_win1(input) and not test_win2(input)

def test_valid_turn_sequence(input):
    # Count the number of moves made by Player 1 and Player 2
    count_1 = input.count('1')
    count_2 = input.count('2')
    
    # Player 2 should never have more moves than Player 1, and the difference should be at most 1
    if not (count_1 == count_2 or count_1 == count_2 + 1):
        return False
    
    # Simulate the game to check for moves after a win
    board = ['0'] * 9
    current_player = '1'
    for _ in range(count_1 + count_2):
        # Find the next move
        for i in range(9):
            if board[i] != input[i]:
                if input[i] != current_player:
                    return False  # Invalid move sequence
                board[i] = current_player
                # Check for a win after this move
                if test_win1(''.join(board)) if current_player == '1' else test_win2(''.join(board)):
                    # Check if any additional moves were made after the win
                    if board != list(input):
                        return False  # Moves made after game has ended
                    else:
                        return True  # Valid sequence with win
                current_player = '2' if current_player == '1' else '1'
                break
    return True  # Valid sequence without any premature wins

def is_valid_win(input, player):
    # Returns True if the player has a winning line and could have formed it on their last move
    counts_p = input.count(player)
    counts_opponent = input.count('1' if player == '2' else '2')
    # Check if it's the player's turn based on move counts
    if player == '1' and counts_p != counts_opponent + 1:
        return False
    if player == '2' and counts_p != counts_opponent:
        return False
    # Use the updated test_valid_turn_sequence to ensure no moves were made after the win
    return test_valid_turn_sequence(input)

def is_valid_win_1(input):
    return is_valid_win(input, '1')

def is_valid_win_2(input):
    return is_valid_win(input, '2')

def solver(input):
    if test_win1(input):
        return 'win1'
    
    if test_win2(input):
        return 'win2'
    
    if test_draw(input):
        return 'draw'
    
    return 'ok'  # Game is still in progress













def generate_solve_results(count=100, helper_functions=[]):
    
    if __name__ == "__main__":
        print('Unique boards:', len(boards))
        print('Valid states:', len(states))
        print('Final states:', len(final_states))
        print()
    
    boards_list = list(boards.keys())
    results = []
    valid = True
    
    while count > 0 and valid:
        count -= 1
        board = random.choice(boards_list)
        expected = boards[board]
        actual = solver(board)
        valid = valid and (expected == actual)
        
        result = {
            'board': board,
            'solver_result': actual,
            'oracle_result': expected,
            'helper_results': {func.__name__: func(board) for func in helper_functions}
        }
        results.append(result)
    
    return results, valid

def format_and_print_results(results, helper_functions=[]):
    labels = ["input", "solver()", "ORACLE_RESPONSE"] + [func.__name__ + "()" for func in helper_functions]
    print(labels)
    
    for r in results:
        formatted_result = [
            r['board'],
            f"solver({r['board']})=={r['solver_result']}",
            f"ORACLE({r['board']},{r['solver_result']})=={'PASS' if r['solver_result'] == r['oracle_result'] else 'FAIL'}"
        ]
        
        for func_name, result in r['helper_results'].items():
            formatted_result.append(f"{func_name}({r['board']})=={'TRUE' if result else 'FALSE'}")
        
        print(formatted_result)

def attempt_solve(count=100, helper_functions=[]):
    results, all_valid = generate_solve_results(count, helper_functions)
    if __name__ == "__main__":
        format_and_print_results(results, helper_functions)
        print(results)
    return results, all_valid

def random_board():
    return random.choice(list(boards.keys()))


def generate_all_answers():
    all_answers = {}
    for board in boards:
        all_answers[board] = oracle(board)
    return all_answers

all_answers = generate_all_answers()

if __name__ == "__main__":
    
    helper_functions = [test_win1, test_win2, test_draw, test_in_progress, test_valid_turn_sequence, is_valid_win_1, is_valid_win_2]


    attempt_solve(100, [])




