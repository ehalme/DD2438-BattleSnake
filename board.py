import typing
import numpy as np
from enum import Enum

class Action(Enum):
    """Helper class for state updating"""
    up = 0
    down = 1
    left = 3
    right = 4

class BoardState(Enum):
    """Helper class for state assignment"""
    free = 0
    food = 1
    hazard = 2
    snake_body = 3
    snake_head = 4

class Board:
    """
    A class to represent the board for fast simulation during search

    TODO: Check if the snake only should decay if the head is in the hazard or of any body part is in the hazard. -edit: will not do decay....
    """    
    def __init__(self, board: typing.Dict, max_health: int, hazard_decay: int, step_decay: int):
        """Disable decay by setting it to a value less than 0"""
        self.height = board["height"]
        self.width = board["width"]
        self.foods = board["food"]
        self.hazards = board["hazards"]
        self.snakes = board["snakes"]
        self.max_health = max_health
        self.hazard_decay = hazard_decay
        self.step_decay = step_decay

        self.observation_buffer = {}
        self.snake_lookup = {k["id"]: v for v, k in enumerate(self.snakes)} # used to retrieve a specific snake based on id fast
        self.who_killed_who = dict()

    def eat_food(self, snake: typing.Dict, pos: typing.Dict) -> None:
        """
        Removes the food at the given position from the board,
        gives the given snake max health and increases its length.
        """
        try:
            self.foods.remove(pos)
            snake['health'] = self.max_health
        except Exception:
            print("Could not eat food at position: ", pos, ", snake: ", snake)

    def kill_snake(self, killer_snake: typing.Dict, killed_snake: typing.Dict) -> None:
        """Removes the snake from the board"""
        try:
            self.snakes.remove(killed_snake)
            self.snake_lookup = {k["id"]: v for v, k in enumerate(self.snakes)}
            self.who_killed_who[killed_snake["id"]] = killer_snake["id"]
        except Exception:
            print("Couldnt kill snake: ", killed_snake)

    def move_snakes(self, moves: typing.Dict[str, Action]) -> None:
        """Takes a dictionary of each snakes (key: snake_id) action and applies them to the board"""
        # Simulate a snake moving in the given direction
        self._reset_observation_buffer()
        
        snakes_to_kill = []
        snakes_to_eat = []
        snakes_to_move = []

        for snake in self.snakes:
            head = snake["head"]
            move = moves[snake["id"]]
            next_head = self._get_next_head(head, move)

            # Get the state of the cell that the head is moving to
            head_states = self.get_cell_state(next_head)

            #print(snake["id"], ", going: ", move, ", to: ", next_head, ", state: ", head_states)

            # Check if the snake is dead
            if head_states is None:
                # Snake died
                snakes_to_kill.append((snake, snake))
                continue

            if BoardState.snake_body in head_states:
                snakes_to_kill.append((head_states[BoardState.snake_body], snake))
                continue

            ate_food = False

            # remove step decay before checking if food is eaten so a snake
            # can eat a piece of food if its on 1hp
            if self.step_decay > 0:
                snake['health'] -= self.step_decay

            if BoardState.food in head_states:
                snakes_to_eat.append((snake, next_head))
                self.eat_food(snake, next_head)
                ate_food = True

            # Not sure if this should be before or after eating food...
            if self.hazard_decay > 0 and BoardState.hazard in head_states: 
                snake['health'] -= self.hazard_decay

            if BoardState.snake_head in head_states:
                # check which snakes wins
                won = self._check_collision(snake, head_states[BoardState.snake_head])
                if won:
                    # Killed snake
                    snakes_to_kill.append((snake, head_states[BoardState.snake_head]))

            if snake['health'] <= 0:
                # Snake died
                snakes_to_kill.append((snake, snake))
                continue

            if not ate_food:
                snakes_to_move.append((snake, next_head))

        for snake, next_head in snakes_to_eat:
            # If we ate food, we just need to move the head
            # since the new body piece will spawn at the tail
            snake['length'] += 1
            # Move the snake
            snake['head'] = next_head
            snake['body'].insert(0, next_head)

        # Kill off the snakes
        for killer, killed in snakes_to_kill:
            self.kill_snake(killer, killed)

        # Move the remaining snakes
        for snake, next_head in snakes_to_move:           
            # Move the snake head and tail, keep same length
            snake['head'] = next_head
            snake['body'][1:] = snake['body'][0:-1]
            snake['body'][0] = next_head


    def get_cell_state(self, position: typing.Dict) -> typing.Dict[BoardState, object]:
        """
        Returns a dict of states of a given position. 
        Returns Dict[BoardState, object] or None if out of bounds.
        """
        # Returns a dict because multiple states can appear on one tile

        # If this is too slow and the game board does not change too much we can create 
        # a matrix as the board state and use that instead. Will result in faster lookup
        # time, but if the game board changes a lot, then the overhead to create
        # the board is worse than these "in" checks
        # -edit: I think using the state cache it will be fast enough.

        # Check cache
        if position in self.observation_buffer:
            return self.observation_buffer[position]
        
        if position['x'] < 0 or position['x'] == self.width:
            return None

        if position['y'] < 0 or position['y'] == self.height:
            return None

        state = dict()

        if position in self.foods:
            state[BoardState.food] = True

        if position in self.hazards:
            state[BoardState.hazard] = True

        for snake in self.snakes:
            if position in snake['body']:
                state[self._get_snake_state(snake, position)] = snake
                break

        if len(state) == 0:
            state[BoardState.free] = True

        # Save cache
        self.observation_buffer[position] = state

        return state

    def get_snake(self, snake_id: str) -> typing.Dict:
        """Returns a snake object (dict) given snake id"""
        if snake_id in self.snake_lookup:
            return self.snakes[self.snake_lookup[snake_id]]

        return None

    def is_snake_alive(self, snake_id: str) -> bool:
        """Returns true if snake is alive"""
        return snake_id in self.snake_lookup

    def get_snake_kills(self, snake_id: str) -> str:
        """Returns the killer of the snake"""
        if snake_id in self.who_killed_who:
            return self.who_killed_who[snake_id]

        return None

    def get_board_matrix(self) -> typing.List[typing.List[typing.Dict[BoardState, object]]]:
        """Returns a 2D matrix representation of the board, origin in top lefthand corner"""
        board = [[0,] * self.height for y in range(self.width)]
        for x in range(self.width):
            for y in range(self.height):
                s = self.get_cell_state({"x": x, "y": y})
                board[x][y] = s

        # it is now origin in top left, rotate to have origin in bottom left

        return board

    def get_board_img(self):
        """Returns an RGB image representation of the board"""
        board = self.get_board_matrix()
        data = np.zeros((self.width, self.height, 3), dtype=np.float32)

        for x in range(self.width):
            for y in range(self.height):
                if BoardState.snake_body in board[x][y]:
                    data[x,y,board[x][y][BoardState.snake_body]["m"]] = 0.8
                elif BoardState.snake_head in board[x][y]:
                    data[x,y,board[x][y][BoardState.snake_head]["m"]] = 1
                elif BoardState.food in board[x][y]:
                    data[x,y,1] = 1
                    data[x,y,0] = 1
                elif BoardState.hazard in board[x][y]:
                    data[x,y,0] = 0.5
                    data[x,y,2] = 0.5

        # Transpose the matrix
        transposed_matrix = np.transpose(data, axes=(1, 0, 2))

        return transposed_matrix

    def _reset_observation_buffer(self):
        self.observation_buffer.clear()

    def _check_collision(self, snake: typing.Dict, other_snake: typing.Dict) -> bool:
        """returns true if snake won"""

        if snake['length'] > other_snake['length']:
            return True
        elif snake['length'] < other_snake['length']:
            return False

        # Else they're equal lenght and both die
        return True

    def _get_next_head(self, head: typing.Dict, move: Action) -> typing.Dict:
        if move == Action.up:
            return {"x": head["x"], "y": head["y"] + 1}
        elif move == Action.down:
            return {"x": head["x"], "y": head["y"] - 1}
        elif move == Action.right:
            return {"x": head["x"] + 1, "y": head["y"]}
        elif move == Action.left:
            return {"x": head["x"] - 1, "y": head["y"]}
        else:
            print("Undefined action!")

    def _get_snake_state(self, snake: str, pos: typing.Dict) -> BoardState:
        # Assumes that the position is somewhere in a snake...      
        if snake['head'] == pos:
            return BoardState.snake_head
        else:
            return BoardState.snake_body