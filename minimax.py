"""
function minimax(node, depth, maximizingPlayer) is
if depth = 0 or node is a terminal node then
    return the heuristic value of node
if maximizingPlayer then
    value := −∞
    for each child of node do
        value := max(value, minimax(child, depth − 1, FALSE))
    return value
else (* minimizing player *)
    value := +∞
    for each child of node do
        value := min(value, minimax(child, depth − 1, TRUE))
    return value
"""

import itertools
import typing
import time
import math

from board import Board, Action
from heuristics import Heuristic


def start_minimax(game_state: typing.Dict, heuristic: Heuristic, max_depth: int, max_health: int, hazard_decay: int, step_decay: int) -> Action:
    """
    Does a minimax run from this starting game_state using the
    specified heuristic.
    Return an Action
    """
    timeout = game_state["game"]["timeout"]
    latency = int(game_state["you"]["latency"]) if game_state["you"]["latency"] != '' else 100
    calculation_time = timeout - latency - 60 # X ms for padding
    calculation_time *= 1e6 # convert ms (10^3) to ns (10^9)
    
    my_snake = game_state['you']['id']
    init_board = Board(game_state["board"], max_health=max_health, hazard_decay=hazard_decay, step_decay=step_decay)

    start_time = time.time_ns()

    friendly_snakes, enemy_snakes, my_snake_alive = get_other_snakes(init_board, my_snake)

    all_friendly = friendly_snakes + [my_snake]
    action_combos = get_children(init_board, all_friendly)

    highest_score = -math.inf
    best_actions = None
    #print(my_snake)
    for actions in action_combos:
        new_board = init_board.copy()

        new_board.move_snakes(actions)
    
        score = minimax(new_board, my_snake, heuristic, calculation_time, start_time, max_depth-1, False, -math.inf, math.inf)

        #print("Score: ", score, ", Actions: ", actions)
        
        if score > highest_score:
            highest_score = score
            best_actions = actions

    print("Minimax time: ", (time.time_ns() - start_time) * 1e-6)

    if best_actions is None:
        return Action.up
    
    return best_actions[my_snake]

    # start the minimax alg from init_board
    # make copies of the board for width and friendly/enemy snakes lists
    # used Board.move_snakes() when going depth
    # use heuristic.get_score(board) to get the score of the current game state

def minimax(board: Board, my_snake: str, heuristic: Heuristic, calculation_time: int, start_time: int, depth: int, maximizing: bool, alpha: int, beta: int) -> float: # TODO: edit object
    friendly_snakes, enemy_snakes, my_snake_alive = get_other_snakes(board, my_snake)
    win = len(enemy_snakes) == 0
    lose = not my_snake_alive # len(friendly_snakes) == 0 and # removing this makes the simulation asymmetric but makes other things easier
    # might want to simulate after the snake is dead too

    used_time = time.time_ns() - start_time
    time_limit_exceeded = used_time > calculation_time
    
    if win or lose or depth == 0 or time_limit_exceeded:
        #print("depth:", depth, ", time limit: ", time_limit_exceeded, ", win: ", win, ", lose: ", lose, ", used_time: ", used_time)
        if maximizing:
            return heuristic.get_score(board, my_snake, friendly_snakes, enemy_snakes)
        else:
            return heuristic.get_score(board, my_snake, enemy_snakes, friendly_snakes)
        
    if maximizing:
        score = -math.inf

        all_friendly = friendly_snakes + [my_snake]
        action_combos = get_children(board, all_friendly)

        for actions in action_combos:
            new_board = board.copy()

            new_board.move_snakes(actions)
            score = max(score, minimax(new_board, my_snake, heuristic, calculation_time, start_time, depth - 1, False, alpha, beta))
            alpha = max(alpha, score)
            if beta <= alpha:
                break

        return score        
    else:
        score = math.inf

        action_combos = get_children(board, enemy_snakes)

        for actions in action_combos:
            new_board = board.copy()

            new_board.move_snakes(actions)
            score = min(score, minimax(new_board, my_snake, heuristic, calculation_time, start_time, depth - 1, True, alpha, beta))
            beta = min(beta, score)
            if beta <= alpha:
                break;
        
        return score
     
def get_children(board: Board, snakes: typing.List[str]):
    snake_actions = [Action.up, Action.down, Action.left, Action.right]
    valid_actions = [[] for i in range(len(snakes))]
    
    for i, snake in enumerate(snakes):
        for action in snake_actions:
            if board.is_valid_action(snake, action): ## if no actions are valid we need to return something too
                valid_actions[i].append(action)
    
    action_combos = list(itertools.product(*valid_actions))
    action_combos = [{snakes[i]: action for i, action in enumerate(actions)} for actions in action_combos]

    
    return action_combos
    
    for action_combo in action_combos:
        invalid = False
        for i, snake in enumerate(snakes):
            if not board.is_valid_action(snake, action_combo[i]): ## if no actions are valid we need to return something too
                invalid = True
                break
        
        if not invalid:
            valid_actions = dict()
            for i, snake in enumerate(snakes):
                valid_actions[snake] = action_combo[i]
            children.append(valid_actions)
    
    return children
     
def get_other_snakes(board: Board, my_snake: str) -> typing.Tuple[typing.List[str], typing.List[str], bool]:
    """
    Returns a list of friendly snakes that are not yourself and a list of all enemy snakes.
    Returns id only.
    """
    friendly_snakes = []
    enemy_snakes = []

    # If my_snake is dead then m_snake will be None
    m_snake, my_snake_alive = board.get_snake(my_snake)

    for snake in board.snakes:
        if snake["name"] != m_snake["name"]:
            enemy_snakes.append(snake["id"])
        elif snake['id'] != m_snake['id']:
            friendly_snakes.append(snake['id'])

    return friendly_snakes, enemy_snakes, my_snake_alive

