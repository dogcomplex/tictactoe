square_state_space = ['0', '1', '2']
label_space = ['ok', 'win1', 'win2', 'draw', 'error']

def tictactoe(input):
    """Standard TicTacToe oracle function"""
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6],              # Diagonals
    ]
    
    for condition in win_conditions:
        if input[condition[0]] == input[condition[1]] == input[condition[2]] == '1':
            return "win1"
    
    for condition in win_conditions:
        if input[condition[0]] == input[condition[1]] == input[condition[2]] == '2':
            return "win2"
    
    if '0' not in input:
        return "draw"
    
    count_1 = input.count('1')
    count_2 = input.count('2')
    if count_1 == count_2:
        return "ok"
    if count_1 != count_2 and count_1 != count_2 + 1:
        return "error"
    
    return "ok"


def tictactoe_no_diags(input):
    """TicTacToe variant where diagonals don't count as wins"""
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        # No diagonals!
    ]
    
    for condition in win_conditions:
        if input[condition[0]] == input[condition[1]] == input[condition[2]] == '1':
            return "win1"
    
    for condition in win_conditions:
        if input[condition[0]] == input[condition[1]] == input[condition[2]] == '2':
            return "win2"
    
    if '0' not in input:
        return "draw"
    
    count_1 = input.count('1')
    count_2 = input.count('2')
    if count_1 == count_2:
        return "ok"
    if count_1 != count_2 and count_1 != count_2 + 1:
        return "error"
    
    return "ok"


oracle = tictactoe

# Generate all valid game states
boards = {}
states = {}
final_states = {}

def move(state):
    if state in states:
        return
    board, moves = state.split('_')
    if oracle(board) == 'error':
        return
    states[state] = oracle(board)
    boards[board] = oracle(board)
    if oracle(board) != 'ok':
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
print(f"TicTacToe all states initialized. Unique boards: {len(boards)}")

import random

def random_board():
    return random.choice(list(boards.keys()))

def generate_all_answers():
    all_answers = {}
    for board in boards:
        all_answers[board] = oracle(board)
    return all_answers

all_answers = generate_all_answers()
