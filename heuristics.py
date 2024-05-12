import typing

class Heuristic:
    def __init__(self, weights: typing.Dict):
        self.weights = weights

    def get_heat_map(self):
        pass
    
    def get_score(self, game_state: typing.Dict):
        # Evaluate the game state based on factors like length, food distance, etc.
        snake_length = len(game_state['you']['body'])
        food_distance = self.calculate_distance_to_food(game_state)
    
        # Assign weights to factors
        length_weight = 1
        food_distance_weight = -1
    
        # Calculate the heuristic score
        score = length_weight * snake_length + food_distance_weight * food_distance
    
        return score
    
    def calculate_distance_to_food(self, game_state: typing.Dict):
        # Calculate the Manhattan distance to the closest food source
        snake_head = game_state['you']['body'][0]
        closest_food = self.find_closest_food(snake_head, game_state['board']['food'])
    
        distance = abs(snake_head['x'] - closest_food['x']) + abs(snake_head['y'] - closest_food['y'])
    
        return distance
    
    def find_closest_food(self, snake_head: typing.Dict, food_positions: typing.Dict):
        # Find the closest food source to the snake's head
        closest_distance = float('inf')
        closest_food = None
    
        for food in food_positions:
                distance = abs(snake_head['x'] - food['x']) + abs(snake_head['y'] - food['y'])
                if distance < closest_distance:
                        closest_distance = distance
                        closest_food = food
    
        return closest_food