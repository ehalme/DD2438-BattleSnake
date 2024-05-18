import time
import numpy as np
import pygame 
import math
from board import Board, Action
from minimax import start_minimax
from heuristics import Heuristic
from weights import weights
from multiprocessing.pool import ThreadPool
from tree_visualizer import get_colour_name

def numpy_array_to_surface(array):
    # Flip the array along the y-axis before creating the surface
    flipped_array = np.flipud(array)
    return pygame.surfarray.make_surface(flipped_array.swapaxes(0, 1))

if __name__ == "__main__":       
    manual_control = False
    fps = 1.2
    max_depth = 30

    heuristic = Heuristic(weights)
    
    boardState = Board(None, max_health=100, hazard_decay=0, step_decay=1, move_all_snakes=True, print_logs=True)
    manual_snake = boardState.snakes[0]
  
    # Scale factor for rendering the image larger
    SCALE_FACTOR = 30

    # Initialize Pygame
    pygame.init()
    screen_width = boardState.width * SCALE_FACTOR * 1.5
    screen_height = boardState.height * SCALE_FACTOR * 1.5 + 50
    image_width = boardState.width * SCALE_FACTOR * 1.5
    image_height = boardState.height * SCALE_FACTOR * 1.5
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Image Rendering")
    pygame.font.init()
    my_font = pygame.font.SysFont('Comic Sans MS', 15)

    # Get teams
    teams = dict()
    for snake in boardState.snakes:
        color_name = get_colour_name(boardState._get_unique_color(snake["m"]))
        if snake["name"] in teams:
            teams[snake["name"]].append(color_name + ", ")
        else:
            teams[snake["name"]] = [color_name + ", ", ]

    # Main loop
    #pool = ThreadPool(len(boardState.snakes))
    pool = ThreadPool(1)
    running = True 
    clock = pygame.time.Clock()
    manual_action = None
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

        
        actions = dict()
        def update_snake(json_board, snake):
            if manual_control and snake["id"] == manual_snake["id"]:
                actions[manual_snake["id"]] = action_manual
                return

            game_state = {
                "game": {"timeout": 500, },
                "board": json_board,
                "you": snake,
            }

            action = start_minimax(game_state, heuristic, max_depth=max_depth, max_health=100, hazard_decay=0, step_decay=1)
            actions[snake["id"]] = action
            #actions = {'totally-unique-snake-id1': Action.up, 'totally-unique-snake-id2': Action.up, 'totally-unique-snake-id3': Action.down}

        json_boards = [boardState.get_json_board() for _ in range(len(boardState.snakes))]
        pool.starmap(update_snake, zip(json_boards, boardState.snakes)) # process all snakes at the same time

        boardState.move_snakes(actions)            
                    
        img = boardState.get_board_img()
        img_surface = numpy_array_to_surface(img)

        # Scale up the image surface
        scaled_img_surface = pygame.transform.scale(img_surface, (image_width, image_height))

        # Blit the scaled image onto the screen
        screen.blit(scaled_img_surface, (0, 0))

        # Write teams to screen
        tm = ""
        for i, team in enumerate(teams):
            tm += f"Team {i}: "
            for color in teams[team]:
                tm += color
            tm = tm[:-2]
            tm += " | "
        tm = tm[:-3]
        teams_surface = my_font.render(tm, False, (255, 255, 255))
        screen.blit(teams_surface, (0,screen_height-25))

        # Update the display
        pygame.display.flip()

        # Cap the frame rate to fps
        clock.tick(fps)

    pygame.quit()
    pool.close()