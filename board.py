import random
import typing
import numpy as np
from enum import Enum
import copy


class Action(Enum):
    """Helper class for state updating"""
    up = 0
    down = 1
    left = 2
    right = 3

class BoardState(Enum):
    """Helper class for state assignment"""
    free = 0
    food = 1
    hazard = 2
    snake_body = 3
    snake_head = 4

class Collision(Enum):
    """Helper class for snake head on collision"""
    won = 0
    lost = 1
    draw = 2

class Board:
    """
    A class to represent the board for fast simulation during search

    TODO: Check if the snake only should decay if the head is in the hazard or of any body part is in the hazard. -edit: will not do decay....
    """    
    def __init__(self, board: typing.Dict | None, max_health: int, hazard_decay: int, step_decay: int, food_spawn_chance=0.15, min_food=1, food_spawn_chances=None, move_all_snakes=False, print_logs=False):
        """
        Disable decay by setting it to a value less than 0.
        Randomly initialize the environment by setting board = None.
        Set the food_spawn_chances to a list of ints in [0-1] so that each environment will use the same food spawn point.
        """
        self.max_health = max_health
        self.hazard_decay = hazard_decay
        self.step_decay = step_decay
        self.food_spawn_chance = food_spawn_chance
        self.food_spawn_chances = food_spawn_chances
        self.min_food = min_food

        self.observation_buffer = dict() # Used to create a sparse matrix of the current game state
        self.print_logs = print_logs

        if board is None:
            self._random_init()
        else:
            self.height = board["height"]
            self.width = board["width"]
            self.foods = board["food"]
            self.hazards = board["hazards"]
            self.snakes = board["snakes"]

        self.snake_lookup = {k["id"]: v for v, k in enumerate(self.snakes)} # used to retrieve a specific snake based on id fast
        self.who_killed_who = dict() # saves which snake killed which snake, stored as snake_id, snake_id
        self.dead_snakes = dict() # all the dead snakes, store as snake_id: True

        self.initial_snake_count = len(self.snakes)
        self.sparse_observation_id_to_row = {k["id"]: v for v, k in enumerate(self.snakes)}

        self.game_step = 0
        self.minor_game_step = 0 # keep track of how many snakes we move and only update game_step when all snakes have moved
        self.move_all_snakes = move_all_snakes # Used when simulating the game, all snakes move at once


    def eat_food(self, snake: typing.Dict, pos: typing.Dict) -> None:
        """
        Removes the food at the given position from the board,
        gives the given snake max health and increases its length.
        """
        try:
            self.foods.remove(pos)
            snake['health'] = self.max_health
        except Exception:
            if self.print_logs:
                print("Could not eat food at position: ", pos, ", snake: ", snake["id"])

    def kill_snake(self, killer_snake: typing.Dict, killed_snake: typing.Dict, reason: str) -> None:
        """Removes the snake from the board"""
        try:
            self.snakes.remove(killed_snake)
            self.snake_lookup = {k["id"]: v for v, k in enumerate(self.snakes)}

            if (self.print_logs):
                print(killer_snake["id"], " killed, ", killed_snake["id"], ", reason: ", reason)
        except Exception:
            if self.print_logs:
                print("Couldnt kill snake: ", killed_snake["id"])

        self.who_killed_who[killed_snake["id"]] = killer_snake["id"]
        self.dead_snakes[killed_snake["id"]] = killed_snake

    def move_snakes(self, moves: typing.Dict[str, Action]) -> None:
        """Takes a dictionary of each snakes (key: snake_id) action and applies them to the board"""
        # Simulate a snake moving in the given direction
        snakes_to_kill = []
        snakes_to_eat = []
        snakes_to_move = []

        for snake_id in moves:
            snake = self.snakes[self.snake_lookup[snake_id]]

            head = snake["head"]
            move = moves[snake["id"]]
            next_head = self._get_next_head(head, move)

            # Get the state of the cell that the head is moving to
            head_states = self.get_cell_state(next_head)

            #print(snake["id"], ", going: ", move, ", to: ", next_head, ", state: ", head_states)

            # Check if the snake is dead
            if head_states is None:
                # Snake died, outside of bounds
                snakes_to_kill.append((snake, snake, "Out of bounds"))
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

            if snake['health'] <= 0:
                # Snake died
                snakes_to_kill.append((snake, snake, "Ran out of health"))
                continue

            if not ate_food:
                snakes_to_move.append((snake, next_head))

        for snake, next_head in snakes_to_eat:
            # If we ate food, we just need to move the head
            # since the new body piece will spawn at the tail
            # Move the snake
            snake['head'] = next_head
            snake['body'].insert(0, next_head)

        # Kill off the snakes
        for killer, killed, reason in snakes_to_kill:
            self.kill_snake(killer, killed, reason)

        # Move the remaining snakes
        for snake, next_head in snakes_to_move:           
            # Move the snake head and tail, keep same length
            snake['head'] = next_head
            snake['body'][1:] = snake['body'][0:-1]
            snake['body'][0] = next_head

        self._reset_observation_buffer() # Reset observation buffer after we moved all objects

        # Check if any snakes should be killed off after they moved...
        snakes_to_kill = []
        for snake_id in moves:
            snake = self.snakes[self.snake_lookup[snake_id]]
            
            head = snake["head"]

            # Get the state of the cell and see if we need to kill any snakes
            head_states = self.get_cell_state(head)

            if BoardState.snake_body in head_states:
                # There can only ever be one snake body part in a cell
                snakes_to_kill.append((head_states[BoardState.snake_body][0], snake, "ran into other snake body"))
                continue

            if head in snake["body"][1:]:
                # Inside of itself
                snakes_to_kill.append((snake, snake, "ran into itself"))
                continue

            if BoardState.snake_head in head_states:
                for other_snake in head_states[BoardState.snake_head]: # This is a list of all snakes in that cell
                    if snake["id"] == other_snake["id"]:
                        # Its just our own phat head
                        continue

                    # check which snakes wins
                    col = self._check_collision(snake, other_snake)
                    if col == Collision.won:
                        # Killed snake
                        snakes_to_kill.append((snake, other_snake, "head on collision"))
                    elif col == Collision.draw:
                        snakes_to_kill.append((snake, other_snake, "traded head on collision"))
                        snakes_to_kill.append((other_snake, snake, "traded head on collision"))

        # Kill off the snakes
        for killer, killed, reason in snakes_to_kill:
            self.kill_snake(killer, killed, reason)

        # Increase length of snakes that ate
        for snake, next_head in snakes_to_eat:
            snake['length'] += 1

        self._reset_observation_buffer() # Reset observation buffer after we moved all objects

        if self.move_all_snakes:
            self.game_step += 1
            self.minor_game_step = self.game_step
        else:
            self.minor_game_step += 0.5
            self.game_step += int(self.minor_game_step)

        # Add food if needed
        add_food = self._check_food_needing_placement()
        if add_food > 0:
            self._place_food_random(add_food)
            self._reset_observation_buffer() # Reset observation buffer after we moved all objects


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
        p = (position["x"], position["y"])
        if p in self.observation_buffer:
            return self.observation_buffer[p]
        
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
                s = self._get_snake_state(snake, position)
                if s in state:
                    state[s].append(snake)
                else:
                    state[s] = [snake,]

        if len(state) == 0:
            state[BoardState.free] = True

        # Save cache
        self.observation_buffer[p] = state

        return state


    def get_snake(self, snake_id: str) -> typing.Tuple[typing.Dict, bool]:
        """Returns a snake object (dict) given snake id and a boolean is_alive"""
        if snake_id in self.snake_lookup:
            return self.snakes[self.snake_lookup[snake_id]], True
        elif snake_id in self.dead_snakes:
            return self.dead_snakes[snake_id], False

        return None, None


    def is_snake_alive(self, snake_id: str) -> bool:
        """Returns true if snake is alive"""
        return snake_id in self.snake_lookup
    

    def is_valid_action(self, snake_id:str, action: Action):
        snake, _ = self.get_snake(snake_id)
        next_pos = self._get_next_head(snake["head"], action)
        cell_state = self.get_cell_state(next_pos)

        # Make sure its inside the bounds
        if cell_state is None:
            return False
        
        # Check if its free
        if BoardState.free in cell_state:
            return True

        # Make sure its not in itself or another snake
        # but allow moves that move onto the tail
        if BoardState.snake_body in cell_state:
            snakes = cell_state[BoardState.snake_body]
            for snake in snakes:
                if snake["body"][-1] != next_pos: # if its not the tail
                    return False
                
        # Allow BoardSate.snake_head

        return True
    

    def get_snake_killer(self, snake_id: str) -> str:
        """Returns the killer of the snake"""
        if snake_id in self.who_killed_who:
            return self.who_killed_who[snake_id]

        return None
    

    def get_sparse_observation(self) -> typing.Tuple:
        obs = [0, ] * (self.initial_snake_count + 1)

        for snake in self.snakes:
            s = [snake["health"],]
            for p in snake["body"]:
                s.append((p["x"], p["y"]))

            obs[self.sparse_observation_id_to_row[snake["id"]]] = tuple(s)

        food_pos = []
        for p in self.foods:
            food_pos.append((p["x"], p["y"]))

        obs[-1] = tuple(food_pos)

        return tuple(obs)


    def get_board_matrix(self) -> typing.List[typing.List[typing.Dict[BoardState, object]]]:
        """Returns a 2D matrix representation of the board, origin in top lefthand corner"""
        self._reset_observation_buffer()
        board = [[0,] * self.height for y in range(self.width)]
        for x in range(self.width):
            for y in range(self.height):
                s = self.get_cell_state({"x": x, "y": y})
                board[x][y] = s

        # it is now origin in top left, rotate to have origin in bottom left

        return board


    def get_board_img(self, background_color=(0,0,0)) -> typing.List[typing.List[typing.List]]:
        """Returns an RGB image representation of the board"""
        board = self.get_board_matrix()
        data = np.zeros((self.width, self.height, 3), dtype=np.float32)

        for x in range(self.width):
            for y in range(self.height):
                if BoardState.snake_body in board[x][y]:
                    color = self._get_unique_color(board[x][y][BoardState.snake_body][0]["m"])
                    data[x,y,0] = color[0] * 0.8
                    data[x,y,1] = color[1] * 0.8
                    data[x,y,2] = color[2] * 0.8
                elif BoardState.snake_head in board[x][y]:
                    color = self._get_unique_color(board[x][y][BoardState.snake_head][0]["m"])
                    data[x,y,0] = color[0]
                    data[x,y,1] = color[1]
                    data[x,y,2] = color[2]
                elif BoardState.food in board[x][y]:
                    data[x,y,1] = 1 * 255
                    data[x,y,0] = 1 * 255
                elif BoardState.hazard in board[x][y]:
                    data[x,y,0] = 0.5 * 255
                    data[x,y,2] = 0.5 * 255
                else:
                    data[x,y,:] = background_color

        # Transpose the matrix
        data = np.transpose(data, axes=(1, 0, 2))

        return data
    

    def print_board(self):
        """Prints the current board to the console"""
        board = self.get_board_matrix()

        snake_color = {self.snakes[i]["id"]: f" {i} " for i in range(len(self.snakes))}

        for x in range(self.width):
            for y in range(self.height):
                if BoardState.snake_body in board[x][y]:
                    board[x][y] = snake_color[board[x][y][BoardState.snake_body][0]["id"]]
                elif BoardState.snake_head in board[x][y]:
                    board[x][y] = " H "
                elif BoardState.food in board[x][y]:
                    board[x][y] = " F "
                elif BoardState.hazard in board[x][y]:
                    board[x][y] = " X "
                else:
                    board[x][y] = " - "
        
        board = np.transpose(board)

        for x in range(self.height-1,-1,-1):
            for y in range(self.width):
                print(board[x][y], end="")
            print("")
    

    def copy(self):
        return copy.deepcopy(self)
    

    def get_json_board(self) -> typing.Dict:
        data = {
                "height": self.height,
                "width": self.width,
                "food": self.foods,
                "hazards": self.hazards,
                "snakes": self.snakes,
            }
        
        return data


    def _reset_observation_buffer(self):
        self.observation_buffer = dict()


    def _check_collision(self, snake: typing.Dict, other_snake: typing.Dict) -> Collision:
        """returns collision event"""

        if snake['length'] > other_snake['length']:
            return Collision.won
        elif snake['length'] < other_snake['length']:
            return Collision.lost

        # Else they're equal lenght and both die
        return Collision.draw


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
        

    def _get_unique_color(self, value) -> typing.Tuple[float,float,float]:
        """Returns a unique color within the range 0-255"""
        import colorsys
        max_value=255
        # Calculate the hue value based on the given number
        hue = (value / max_value) * 360
        # Convert HSL to RGB
        rgb = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
        # Convert RGB values from floats in the range 0.0 - 1.0 to integers in the range 0 - 255
        rgb = tuple(int(x * 255) for x in rgb)
        return rgb
    

    def _random_init(self) -> None:
        self.height = 11
        self.width = 11
        self.snakes = self._get_snakes(number_of_snakes=4, snake_start_size=3)
        self.hazards = []
        self.foods = []
        self._place_food_random(len(self.snakes))
    

    def _place_food_random(self, number_of_food) -> None:
        free_positions = self._get_free_positions()

        if len(free_positions) < number_of_food:
            return

        if self.print_logs:
            print("Placing: ", number_of_food, " foods on the board!")        

        if self.food_spawn_chances is not None:
            random.seed(self.food_spawn_chances[self.game_step%len(self.food_spawn_chances)])
        
        self.foods.extend(random.sample(free_positions, number_of_food))
    

    def _check_food_needing_placement(self) -> int:
        if self.minor_game_step != self.game_step: # only update on full game steps
            return 0

        num_current_food = len(self.foods)

        if num_current_food < self.min_food:
            return self.min_food - num_current_food

        if self.food_spawn_chances is not None:
            if self.food_spawn_chances[self.game_step%len(self.food_spawn_chances)] < self.food_spawn_chance:
                return 1
        elif self.food_spawn_chance > 0 and random.random() < self.food_spawn_chance:
            return 1
        
        return 0


    def _get_free_positions(self) -> typing.List[typing.Dict]:
        free_positions = []
        for x in range(self.width):
            for y in range(self.height):
                p = {"x": x, "y": y}
                s = self.get_cell_state(p)
                if BoardState.free in s:
                    free_positions.append(p)

        return free_positions
    

    def _get_snakes(self, number_of_snakes: int, snake_start_size: int) -> typing.List[typing.Dict]:
        class RandomPositionBucket:
            """Just a helper class"""
            def __init__(self):
                self.positions = []

            def fill(self, *points):
                for point in points:
                    self.positions.append(point)

            def take(self):
                if not self.positions:
                    raise Exception("No positions left to take")
                return self.positions.pop(random.randint(0, len(self.positions) - 1))
            
        bodies = [[] for i in range(number_of_snakes)]

        quad_h_space = self.width // 2
        quad_v_space = self.height // 2

        h_offset = quad_h_space // 3
        v_offset = quad_v_space // 3

        quads = [RandomPositionBucket() for _ in range(4)]

        # quad 1
        quads[0].fill(
            (h_offset, v_offset),
            (quad_h_space - h_offset, v_offset),
            (h_offset, quad_v_space - v_offset),
            (quad_h_space - h_offset, quad_v_space - v_offset),
        )

        # quad 2
        for p in quads[0].positions:
            quads[1].fill((self.width - p[0] - 1, p[1]))

        # quad 3
        for p in quads[0].positions:
            quads[2].fill((p[0], self.height - p[1] - 1))

        # quad 4
        for p in quads[0].positions:
            quads[3].fill((self.width - p[0] - 1, self.height - p[1] - 1))

        current_quad = random.randint(0, 3)  # randomly pick a quadrant to start from
        # evenly distribute snakes across quadrants, randomly, by rotating through the quadrants
        for i in range(number_of_snakes):
            p = quads[current_quad].take()
            for _ in range(snake_start_size):
                bodies[i].append({"x": p[0], "y": p[1]})
            current_quad = (current_quad + 1) % 4

        snakes = [self._create_snake(f"id{i}", i, (i+1)/number_of_snakes * 255, body) for i, body in enumerate(bodies)]

        return snakes
    
    def _create_snake(self, snake_id: str, id_number: int, color: int, body: typing.List[typing.Dict]):
        return {
            "id": snake_id,
            "m": color,
            "name": "Sneky McSnek Face " + str(id_number%2),
            "health": self.max_health,
            "body": body,
            "latency": 50,
            "head": body[0],
            "length": len(body),
            "shout": "why are we shouting??",
            "squad": "1",
            "customizations":{
                "color":"#26CF04",
                "head":"smile",
                "tail":"bolt"
            }
        }
