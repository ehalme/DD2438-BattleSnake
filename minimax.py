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

import typing
import time

from board import Board, BoardState, Action
from heuristics import Heuristic


def start_minimax(game_state: typing.Dict, heuristic: Heuristic, max_health: int, hazard_decay: int, step_decay: int) -> Action:
    """
    Does a minimax run from this starting game_state using the
    specified heuristic.
    Return an Action
    """
    timeout = game_state["game"]["timeout"]
    latency = game_state["you"]["latency"]
    calculation_time = timeout - latency - 50 # 50 ms for padding
    calculation_time *= 1e6 # convert ms (10^3) to ns (10^9)
    
    my_snake = game_state['you']['id']
    init_board = Board(game_state["board"], max_health=max_health, hazard_decay=hazard_decay, step_decay=step_decay)

    #
    minimax(init_board, my_snake, heuristic)

    # start the minimax alg from init_board
    # make copies of the board for width and friendly/enemy snakes lists
    # used Board.move_snakes() when going depth
    # use heuristic.get_score(board) to get the score of the current game state

def minimax(board: Board, my_snake: str, heuristic: Heuristic) -> object: # TODO: edit object
    friendly_snakes, enemy_snakes = get_other_snakes(board, my_snake)
    win = len(enemy_snakes) == 0
    lose = len(friendly_snakes) == 0 and not init_board.is_snake_alive(my_snake)
    start_time = time.perf_counter_ns()
    time_limit = time.perf_counter_ns() - start_time < calculation_time
    
    if win or lose or depth == max_depth:
        
    

def get_other_snakes(board: Board, my_snake: str) -> (typing.List[str], typing.List[str]):
    """
    Returns a list of friendly snakes that are not yourself and a list of all enemy snakes.
    Returns id only.
    """
    friendly_snakes = []
    enemy_snakes = []

    m_snake = board.get_snake(my_snake)

    for snake in board.snakes:
        if snake["name"] != m_snake["name"]:
            enemy_snakes.append(snake["id"])
        elif snake['id'] != m_snake['id']:
            friendly_snakes.append(snake['id'])

    return friendly_snakes, enemy_snakes

