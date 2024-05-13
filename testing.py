import time
import numpy as np
import pygame 
import math
from board import Board, Action
from minimax import start_minimax
from heuristics import Heuristic

def numpy_array_to_surface(array):
    # Flip the array along the y-axis before creating the surface
    flipped_array = np.flipud(array)
    return pygame.surfarray.make_surface(flipped_array.swapaxes(0, 1))


if __name__ == "__main__":   
    s1 = {
            "id": "totally-unique-snake-id1",
            "m": 0,
            "name": "Sneky McSnek Face1",
            "health": 54,
            "body": [
                {"x": 0, "y": 0},
                {"x": 1, "y": 0},
                {"x": 2, "y": 0}
            ],
            "latency": 123,
            "head": {"x": 0, "y": 0},
            "length": 3,
            "shout": "why are we shouting??",
            "squad": "1",
            "customizations":{
                "color":"#26CF04",
                "head":"smile",
                "tail":"bolt"
            }
        }

    s2 = {
            "id": "totally-unique-snake-id2",
            "m": 1,
            "name": "Sneky McSnek Face1",
            "health": 54,
            "body": [
                {"x": 5, "y": 0},
                {"x": 6, "y": 0},
                {"x": 7, "y": 0}
            ],
            "latency": 123,
            "head": {"x": 5, "y": 0},
            "length": 3,
            "shout": "why are we shouting??",
            "squad": "1",
            "customizations":{
                "color":"#26CF04",
                "head":"smile",
                "tail":"bolt"
            }
        }

    s3 = {
            "id": "totally-unique-snake-id3",
            "name": "Sneky McSnek Face2",
            "m": 2,
            "health": 54,
            "body": [
                {"x": 5, "y": 1},
                {"x": 6, "y": 1},
                {"x": 6, "y": 2}
            ],
            "latency": 123,
            "head": {"x": 5, "y": 1},
            "length": 3,
            "shout": "why are we shouting??",
            "squad": "1",
            "customizations":{
                "color":"#26CF04",
                "head":"smile",
                "tail":"bolt"
            }
        }
    
    s4 = {
            "id": "totally-unique-snake-id4",
            "name": "Sneky McSnek Face3",
            "m": 2,
            "health": 54,
            "body": [
                {"x": 6, "y": 8},
                {"x": 7, "y": 8},
                {"x": 8, "y": 8}
            ],
            "latency": 123,
            "head": {"x": 6, "y": 8},
            "length": 3,
            "shout": "why are we shouting??",
            "squad": "1",
            "customizations":{
                "color":"#26CF04",
                "head":"smile",
                "tail":"bolt"
            }
        }

    b_example = {
                "height": 11,
                "width": 13,
                "food": [
                    {"x": 5, "y": 5},
                    {"x": 9, "y": 0},
                    {"x": 12, "y": 10}
                ],
                "hazards": [
                    #{"x": 0, "y": 1},
                    #{"x": 0, "y": 2}
                ],
                "snakes": [
                    s1, s2, s3, s4
                ]
            }

    game_state = {
        "game": {"timeout": 500, },
        "board": b_example,
        "you": s1,
    }

    weights = {
            "food_distance": 1,
            "enemy_distance": -0.5,
            "friendly_distance": 0,
            "death": -10,
            "enemy_killed": 1,
            "friendly_killed": -1,
        }
    
    manual_control = False
    fps = 3
    max_depth = 6

    manual_snake = s1
    heuristic = Heuristic(weights)
    
    boardState = Board(b_example, max_health=100, hazard_decay=2, step_decay=1) # temp
  
    # Scale factor for rendering the image larger
    SCALE_FACTOR = 30

    # Initialize Pygame
    pygame.init()
    screen_width = b_example["width"] * SCALE_FACTOR
    screen_height = b_example["height"] * SCALE_FACTOR
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Image Rendering")

    # Load your numpy array as an image
    # Assuming img_array is your numpy array
    

    # Main loop
    running = True
    last_update = time.time() + 1
    manual_action = None
    actions = dict()
    action_manual = Action.up
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Check for key presses
                if event.key == pygame.K_w:
                    action_manual = Action.up
                elif event.key == pygame.K_s:
                    action_manual = Action.down
                elif event.key == pygame.K_d:
                    action_manual = Action.right
                elif event.key == pygame.K_a:
                    action_manual = Action.left                    

        if time.time() - last_update > 1/fps:
            json_board = boardState.get_json_board()
            for snake in boardState.snakes:
                if manual_control and snake["id"] == manual_snake["id"]:
                    actions[manual_snake["id"]] = action_manual
                    continue

                game_state = {
                    "game": {"timeout": 500, },
                    "board": json_board,
                    "you": snake,
                }

                action = start_minimax(game_state, heuristic, max_depth=max_depth, max_health=100, hazard_decay=0, step_decay=1)
                actions[snake["id"]] = action
                #actions = {'totally-unique-snake-id1': Action.up, 'totally-unique-snake-id2': Action.up, 'totally-unique-snake-id3': Action.down}
            
            boardState.move_snakes(actions)
            last_update = time.time()
            
                    
        img = boardState.get_board_img()
        img_surface = numpy_array_to_surface(img)

        # Scale up the image surface
        scaled_img_surface = pygame.transform.scale(img_surface, (screen_width, screen_height))

        # Blit the scaled image onto the screen
        screen.blit(scaled_img_surface, (0, 0))

        # Update the display
        pygame.display.flip()

    pygame.quit()