import typing
from board import Board
from collections import deque

class Heuristic:
    def __init__(self, weights: typing.Dict):
        """
        Weights are a dictionary of the form:
        {
            "food_distance": X,
            "enemy_distance": X,
            "friendly_distance": X,
            "death": X,
            "enemy_killed": X,
            "friendly_killed": X,
            ...
        }
        """
        self.weights = weights
    
    def get_score(self, board: Board, my_snake: str, friendly_snakes: typing.List[str], enemy_snakes: typing.List[str]) -> float:
        """
        Evaluates a game state and returns a score given the current board, my snake, friendly snakes and the other snakes.
        board: Board object
        my_snake: my snake id
        friendly_snakes: list of friendly snake ids
        enemys_snakes: list of enemy snake ids
        """
        score = 0

        # Closest food
        food_distances = self.calculate_distances_to_food(board, my_snake)
        closest_food_point, closest_food_distance = self.find_closest_point(food_distances)

        if closest_food_distance is not None:
            score += self.weights["food_distance"] * 1/(closest_food_distance + 1)
        """
        # Closest enemy
        enemy_distances = self.calculate_distance_to_snakes(board, my_snake, enemy_snakes)
        closest_enemy_point, closest_enemy_distance = self.find_closest_point(enemy_distances)

        if closest_enemy_distance is not None:
            score += self.weights["enemy_distance"] * 1/(closest_enemy_distance + 1)

        # Closest friendly 
        friendly_distances = self.calculate_distance_to_snakes(board, my_snake, friendly_snakes)
        closest_friendly_point, closest_friendly_distance = self.find_closest_point(friendly_distances)

        if closest_friendly_distance is not None:
            score += self.weights["friendly_distance"] * 1/(closest_friendly_distance + 1)
        """
        # Did we kill enemies?
        killed_enemies = self.calculate_killed_snakes(board, my_snake, enemy_snakes)
        score += self.weights["enemy_killed"] * killed_enemies / 2

        # Did we kill friendlies? :(
        killed_friendly = self.calculate_killed_snakes(board, my_snake, friendly_snakes)
        score += self.weights["friendly_killed"] * killed_friendly / 2

        # Are we alive? :O
        my_snake_dict, is_alive = board.get_snake(my_snake)
        if is_alive is not None:
            score += self.weights["death"] * int(not is_alive)

        if my_snake_dict is None:
            return score
        
        # Health reward
        current_health = my_snake_dict["health"]
        score += self.weights["health"] * current_health / board.max_health

        # Eat food reward
        if my_snake_dict["head"] in board.foods:
            score += self.weights["eat_food"]

        # Reachable cells rewards
        board_size = board.width * board.height
        reachable_cells = self.flood_fill(board, my_snake)
        score += self.weights["flood_fill"] * reachable_cells / board_size

        length = my_snake_dict["length"]
        score += self.weights["length"] * length / board_size
    
        return score

    def calculate_killed_snakes(self, board: Board, my_snake: str, other_snakes: typing.List[str]) -> int:
        if other_snakes is None or len(other_snakes) < 1:
            return 0
        
        killed = 0        
        for snake in other_snakes:
            killer = board.get_snake_killer(snake)
            if killer is not None and killer == my_snake:
                killed += 1

        return killed
    
    def calculate_distances_to_food(self, board: Board, my_snake: str) -> typing.Dict:
        """Calculate the Manhattan distance to the all the food sources"""
        if board.foods is None or len(board.foods) < 1:
            return None
        
        distances = {}
        
        m_snake, _ = board.get_snake(my_snake)
        snake_head = m_snake["head"]
        
        for food in board.foods:
            # key: point on board, value: distance from snake head to food
            p = (food["x"], food["y"])
            distances[p] = self.distance_metric(snake_head, food)
        
        return distances

    def calculate_distance_to_snakes(self, board: Board, my_snake: str, other_snakes: typing.List[str]) -> typing.Dict[typing.Tuple[int,int], float]:
        if other_snakes is None or len(other_snakes) < 1:
            return None
            
        distances = {}
        
        m_snake, _ = board.get_snake(my_snake)
        m_snake_head = m_snake["head"]
        
        for o in other_snakes:
            if o == my_snake:
                continue

            o_snake, _ = board.get_snake(o)
            dists = {}
            for pos in o_snake["body"]:
                p = (pos["x"], pos["y"])
                dists[p] = self.distance_metric(m_snake_head, pos)

            closest_point, closest_distance = self.find_closest_point(dists)
            distances[closest_point] = closest_distance

        return distances
    
    def find_closest_point(self, distances: typing.Dict) -> typing.Tuple[typing.Dict, float]:
        """
        Returns the closest point in the given distance dictionary.
        distances has the strucutre: 
            key: point on board, value: distance from point on board to reference point
        """
        if distances is None or len(distances) < 1:
            return None, None
        
        closest_distance = float('inf')
        closest_point = None
    
        for point in distances:
            if distances[point] < closest_distance:
                closest_distance = distances[point]
                closest_point = point
    
        return closest_point, closest_distance

    def distance_metric(self, point1: typing.Dict, point2: typing.Dict) -> float:
        return abs(point1['x'] - point2['x']) + abs(point1['y'] - point2['y'])

    def flood_fill(self, board: Board, my_snake_id: str) -> int:
        visited = [[False for x in range(board.width)] for y in range(board.height)]
        my_snake, _ = board.get_snake(my_snake_id)
        my_snake_head_x = my_snake["head"]["x"]
        my_snake_head_y = my_snake["head"]["y"]
        snakes = board.snakes
        for snake in snakes:
            for snake_body in snake["body"]:
                visited[snake_body["x"]][snake_body["y"]] = True
        visited[my_snake_head_x][my_snake_head_y] = False

        queue = deque([(my_snake_head_x, my_snake_head_y)])
        reachable_cells = 0

        while queue:
            x, y = queue.popleft()
            if 0 <= x < board.width and 0 <= y < board.height and not visited[x][y]:
                visited[x][y] = True
                reachable_cells += 1
                queue.append((x+1, y))
                queue.append((x-1, y))
                queue.append((x, y+1))
                queue.append((x, y-1))

        return reachable_cells