from environments.connect_four import ConnectFourState, ConnectFour
import numpy as np
import time

# Written By Noah Robertson

def evaluate_row(row, player):
    EMPTY_SPACE = 0
    score = 0
    best_move = -1
    consecutive = 0 
    
    # Iterate through the line to evaluate score based on patterns
    for i in range(len(row)):
        if row[i] == player:
            consecutive += 1
        else:
            if consecutive > 0:
                if i < len(row) and row[i] == EMPTY_SPACE: 
                    # Adjust score based on the sequence length
                    if consecutive >= 3:
                        return 100, i  # Immediate win condition
                    elif consecutive == 2:
                        score += 20
                        best_move = max(best_move, i, key=lambda x: abs(score))
                    elif consecutive == 1:
                        score += 5
                        best_move = max(best_move, i, key=lambda x: abs(score))
                consecutive = 0 
                
            if row[i] == -player:
                score -= 1
                
    # Final check if the line ends with a sequence
    if consecutive > 0 and best_move == -1:
        best_move = len(row)
    
    return score, best_move

def evaluate_col(line, player):
    EMPTY_SPACE = 0
    score = 0
    prev_i = 0
    two_in_row = False
    three_in_row = False
    it = 0
    for i in range(len(line) - 1, 0, -1):
        prev_score = score
        prev_it = it
        #If There is a three in a row and the is the players and the next space is a empty space, give it maximum score
        if line[i] == EMPTY_SPACE and prev_i == player and three_in_row:
            return 100, i
        #If There is a three in a row and the is the opponents and the next space is a empty space, give it minumum score
        elif line[i] == EMPTY_SPACE and prev_i == -player and three_in_row:
            return -100, i
        #Reset if player is blocking any three in a row
        elif line[i] == -player and prev_i == player and three_in_row:
            score = -1
            three_in_row = True
            prev_i = -player
            it = i

        elif line[i] == player and prev_i == -player and three_in_row:
            score = 1
            three_in_row = True
            prev_i = player
            it = i

        #If there is a two in a row and it is the players and the next space is empty, give it a greater score
        elif line[i] == EMPTY_SPACE and prev_i == player and two_in_row:
            score += 20
            prev_i = EMPTY_SPACE
            it = i

        #If there is a two in a row and it is the opponents and the next space is empty, give it a lower score
        elif line[i] == EMPTY_SPACE and prev_i == -player and two_in_row:
            score -= 20
            three_in_row = True
            prev_i = EMPTY_SPACE
            it = i
        
        #Reset if player is blocking
        elif line[i] == -player and prev_i == player and two_in_row:
            score = -1
            prev_i = -player
            it = i

        elif line[i] == player and prev_i == -player and two_in_row:
            score = 1
            prev_i = player
            it = i
        #Tracks if there is going to be a player's three in a row and adds 10 points
        elif line[i] == player and prev_i == player and two_in_row:
            score += 10
            three_in_row = True
            prev_i = player
            it = i

        #Tracks if there is going to be a opponents's three in a row and subtracts 10 points
        elif line[i] == -player and prev_i == -player and two_in_row:
            score -= 10
            three_in_row = True
            prev_i = -player
            it = i
        #Tracks if there is going to be a player's two in a row and adds 5 from the score for a players two in a row
        elif line[i] == player and prev_i == player:
            score += 5
            two_in_row = True
            prev_i = player
            it = i
        #Tracks if there is going to be a opponents's two in a row and subtracts 5 from the score for a opponents two in a row
        elif line[i] == -player and prev_i == -player:
            score -= 5
            two_in_row = True
            prev_i = -player
            it = i
        #If there is a player piece, add 1 point to score then track it
        elif line[i] == player:
            score += 1
            prev_i = player
            it = i
        #If there is a opponent piece, subtract 1 point to score then track it
        elif line[i] == -player:
            score -= 1
            prev_i = -player
            it = i
        #If there is nothing left to track (AKA it is two back to back empty spaces) - break.
        if (prev_score > 0 and prev_score > score):
            it = prev_it
            score = prev_score
        if (prev_score < 0 and prev_score < score):
            it = prev_it
            score = prev_score
    return score, it

#Only accounts for rows and columns
def heuristic_function(state, env, player):
    #Eval = sigma(n, i=0) w_i - sigma(n', i=0)w_i
    #https://www.scirp.org/journal/paperinformation?paperid=125554
    """
    Board Layout
    [0,0] - [0,6]
    ...      ...
    [6,0] - [6,6]
    For row: evaluate horizontal
    For col: evaluate vertical
    For every spot: evaluate diagonal
    """
    board = state.grid

    row_score = []
    col_score = []
    #diag_score = {}


    for row in range(board.shape[0]):
        row_score.append(evaluate_row(board[row], player))
        

    for col in range(board.shape[1]):
        col_score.append(evaluate_col(board.transpose()[col], player))

    row_max_tuple = max(row_score, key=lambda x: x[0])
    row_max = row_max_tuple[0]
    row_max_idx = row_max_tuple[1]

    row_min_tuple = min(row_score, key=lambda x: x[0])
    row_min = row_min_tuple[0]
    row_min_idx = row_min_tuple[1]

    if(abs(row_max) > abs(row_min)):
        row_value = row_max
        row_idx = row_max_idx
    else:
        row_value = row_min
        row_idx = row_min_idx

    col_max_tuple = max(col_score, key=lambda x: x[0])
    col_max = col_max_tuple[0]
    col_max_idx = col_max_tuple[1]

    col_min_tuple = min(col_score, key=lambda x: x[0])
    col_min = col_min_tuple[0]
    col_min_idx = col_min_tuple[1]

    if(abs(col_max) > abs(col_min)):
        col_value = col_max
        col_idx = col_max_idx
    else:
        col_value = col_min
        col_idx = col_min_idx    

    """print("New Iteration")
    print("")
    print("Player: ")
    print(player)
    print("")
    print(board)
    print("--------Board--------")
    print(row_score)
    print("--------Row Score--------")
    print(col_score)
    print("--------Col Score-------")"""

    if(abs(row_value) >= abs(col_value)):
        #print(row_value, row_idx)
        #print("--------Row Selected---------")
        return (row_value, row_idx)
    else:
        #print (col_value, col_idx)
        #print("--------Col Selected---------")
        return (col_value, col_idx)

     
#Based of Minimax Search Pseudo Code
def make_move(state: ConnectFourState, env: ConnectFour) -> int:
    """

    :param state: the current state
    :param env: the environment
    :return: the action to take
    """
    #player = env.next_state(state)
    _, move = max_value(state,env,4)
    return move

def max_value(state: ConnectFourState, env: ConnectFour, depth: int):
    if env.is_terminal(state) or depth <= 0:
            if (depth > 0):
                return (env.utility(state), None)
            return heuristic_function(state, env, 1)
    v = float('-inf')
    for action in env.get_actions(state):
        v2, _ = min_value(env.next_state(state, action), env, depth - 1)
        if (v2 > v):
            v, move = v2, action
    return (v, move)

def min_value(state: ConnectFourState, env: ConnectFour, depth: int):
    if env.is_terminal(state) or depth <= 0:
            if (depth > 0):
                return (env.utility(state), None)
            return heuristic_function(state, env, -1)
    v = float('inf')
    for action in env.get_actions(state):
        v2, _ = max_value(env.next_state(state, action), env, depth - 1)
        if (v2 < v):
            v, move = v2, action
    return (v, move)