import copy
import random
import time
import numpy as np
import pygame 
import math
import typing
from board import Board, Action
from minimax import get_children, get_other_snakes
from heuristics import Heuristic
from weights import weights
import webcolors
from custom_game_state import game_state as custom_game_state

"""This is a really hacky and poorly written visualizer of our minimax algorithm"""

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)

    return closest_name

images = dict()
pruned_branches = 0

def numpy_array_to_surface(array):
    # Flip the array along the y-axis before creating the surface
    flipped_array = np.flipud(array)
    return pygame.surfarray.make_surface(flipped_array.swapaxes(0, 1))

def start_minimax(game_state: typing.Dict, heuristic: Heuristic, max_depth: int, max_health: int, hazard_decay: int, step_decay: int) -> Action:
    """
    Does a minimax run from this starting game_state using the
    specified heuristic.
    Return an Action
    """
    timeout = game_state["game"]["timeout"]
    latency = int(game_state["you"]["latency"]) if game_state["you"]["latency"] != '' else 100
    calculation_time = timeout - latency - 50 # 50 ms for padding
    calculation_time = 50000000
    calculation_time *= 1e6 # convert ms (10^3) to ns (10^9)
    
    my_snake = game_state['you']['id']
    init_board = Board(game_state["board"], max_health=max_health, hazard_decay=hazard_decay, step_decay=step_decay)
    # Disable food spawning for tree visuals
    init_board.food_spawn_chance = 0
    init_board.min_food = 0

    start_time = time.time_ns()

    friendly_snakes, enemy_snakes, my_snake_alive = get_other_snakes(init_board, my_snake)

    all_friendly = friendly_snakes + [my_snake]
    action_combos = get_children(init_board, all_friendly)

    transposition_table = dict()
    highest_score = -math.inf
    best_actions = None
    for depth in range(1,max_depth+1):
        for i, actions in enumerate(action_combos):
            new_board = init_board.copy()

            new_board.move_snakes(actions)

            score = minimax(new_board, my_snake, heuristic, calculation_time, start_time, max_depth-1, False, 1+0.001*(i+1), -math.inf, math.inf, transposition_table)

            #print("Score: ", score, ", Actions: ", actions)
            
            if (max_depth) in images:
                images[max_depth].append((actions, score, new_board.get_board_img(snake_background_color), 0, 1+0.001*(i+1)))
            else:
                images[max_depth] = [(actions, score, new_board.get_board_img(snake_background_color), 0, 1+0.001*(i+1)),]

            if score > highest_score:
                highest_score = score
                best_actions = actions

        used_time = time.time_ns() - start_time
        if used_time >= calculation_time:
            break

    print("Minimax time: ", (time.time_ns() - start_time) * 1e-6)

    if best_actions is None:
        return Action.up
    
    return best_actions[my_snake]

    # start the minimax alg from init_board
    # make copies of the board for width and friendly/enemy snakes lists
    # used Board.move_snakes() when going depth
    # use heuristic.get_score(board) to get the score of the current game state

def minimax(board: Board, my_snake: str, heuristic: Heuristic, calculation_time: int, start_time: int, depth: int, maximizing: bool, parent: int, alpha: int, beta: int, transposition_table: typing.Dict) -> float: # TODO: edit object
    observation = board.get_sparse_observation()
    if observation in transposition_table:
        return transposition_table[observation]
    
    global pruned_branches
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

        for i, actions in enumerate(action_combos):
            new_board = board.copy()

            new_board.move_snakes(actions)
            score = max(score, minimax(new_board, my_snake, heuristic, calculation_time, start_time, depth - 1, False, parent+1+0.001*(i+1), alpha, beta, transposition_table))
            if depth in images:
                if not (actions in images[depth]):
                    images[depth].append((actions, score, new_board.get_board_img(snake_background_color), parent, parent+1+0.001*(i+1)))
            else:
                images[depth] = [(actions, score, new_board.get_board_img(snake_background_color), parent, parent+1+0.001*(i+1)),]
            alpha = max(alpha, score)
            if beta <= alpha:
                pruned_branches += len(action_combos) - i - 1
                transposition_table[observation] = score
                break        
    else:
        score = math.inf

        action_combos = get_children(board, enemy_snakes)

        for i, actions in enumerate(action_combos):
            new_board = board.copy()

            new_board.move_snakes(actions)
            score = min(score, minimax(new_board, my_snake, heuristic, calculation_time, start_time, depth - 1, True, parent+1+0.00001*(i+1), alpha, beta, transposition_table))
            if depth in images:
                if not (actions in images[depth]):
                    images[depth].append((actions, score, new_board.get_board_img(snake_background_color), parent, parent+1+0.00001*(i+1)))
            else:
                images[depth] = [(actions, score, new_board.get_board_img(snake_background_color), parent, parent+1+0.00001*(i+1)),]
            beta = min(beta, score)
            if beta <= alpha:
                pruned_branches += len(action_combos) - i - 1
                transposition_table[observation] = score
                break

    #transposition_table[observation] = score    
    return score

if __name__ == "__main__":       
    # Screen size
    screen_width = 2000
    screen_height = 1200
    # Scale factor for rendering the image larger (game snapshots)
    SCALE_FACTOR = 10
    # Depth to run minimax
    max_depth = 3
    # Split images from the same depth into multiple rows
    split_depths = True
    # How often to draw
    fps = 60
    # Line between node opacity (0-255)
    line_opacity = 255
    # Tree separations
    image_x_separation = 60
    image_y_separation = 120 
    text_y_separation = 15
    # Font size
    font_size = 12
    # Set to true creates a black background
    invert_colors = False
    # Display which graph is drawn on, increase for larger depths (max ~50_000). Zooming in is slow on large images
    large_width = 10_000
    large_height = 10_000
    # Set seed
    random.seed(0)
    # Remove duplicate states
    remove_duplicate_states = True
    # Custom game state
    use_predefined_game_state = False

    ####################################################################################################################
    snake_background_color = (255,255,255) if invert_colors else (0,0,0)

    heuristic = Heuristic(weights)
    
    if use_predefined_game_state:
        boardState = Board(custom_game_state["board"], max_health=100, hazard_decay=2, step_decay=1, print_logs=True)
        game_state = custom_game_state
    else:
        boardState = Board(None, max_health=100, hazard_decay=2, step_decay=1, print_logs=True)
        game_state = {
                        "game": {"timeout": 500, },
                        "board": boardState.get_json_board(),
                        "you": boardState.snakes[0],
                    }

    boardState.food_spawn_chance = 0
    boardState.min_food = 0
    initial_node_im = boardState.get_board_img(snake_background_color)

    # Initialize Pygame
    pygame.init()
    image_width = boardState.width * SCALE_FACTOR
    image_height = boardState.height * SCALE_FACTOR
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.HWSURFACE)
    pygame.display.set_caption("Image Rendering")
    pygame.font.init()
    my_font = pygame.font.SysFont('Comic Sans MS', font_size)

    # Large surface to render tree on
    large_surface = pygame.Surface((large_width, large_height), pygame.SRCALPHA | pygame.HWSURFACE)

    # Surface to draw lines on
    #line_surface = pygame.Surface(large_surface.get_size(), pygame.SRCALPHA)

    # Initialize variables for panning
    zoom_scale = 0.4
    starting_y = 1000
    view_x = -large_width/2 * zoom_scale + screen_width/2
    view_y = -starting_y * zoom_scale + 80
    is_panning = False
    pan_start = (0, 0)

    action = start_minimax(game_state, heuristic, max_depth=max_depth, max_health=100, hazard_decay=0, step_decay=1)
    print("Best action: ", action.name)

    # images = (actions, score, new_board.get_board_img(), parent, self)
    total_number_of_images = 1 # we insert top node later
    ims = [[] for i in range(len(images))]
    for depth in images:
        for img in images[depth]:
            ims[(max_depth) - depth].append(img)
            total_number_of_images += 1

    ims.insert(0,[([], 0, initial_node_im, -1, 0),])

    unique_parents = set()
    for depth in ims:
        for data in depth:
            unique_parents.add(data[3])

    unique_parents = len(unique_parents)

    # Main loop
    running = True
    drawn = False
    clock = pygame.time.Clock()
    last_zoom = 0
    scaled_image = None
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    is_panning = True
                    pan_start = event.pos
                elif event.button == 4:  # Scroll up for zoom in
                    zoom_scale *= 1.1
                    view_x -= (screen_width / 2 - view_x) * 0.1
                    view_y -= (screen_height / 2 - view_y) * 0.1
                elif event.button == 5:  # Scroll down for zoom out
                    zoom_scale /= 1.1
                    # Ensure the scale factor doesn't go too small
                    zoom_scale = max(0.1, zoom_scale)
                    # Adjust the view position to zoom out from the center of the screen
                    if zoom_scale != 0.1:
                        view_x += (screen_width / 2 - view_x) * (1-1/1.1)
                        view_y += (screen_height / 2 - view_y) * (1-1/1.1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    is_panning = False

        if not drawn:
            if invert_colors:
                large_surface.fill((0, 0, 0))
            else:
                large_surface.fill((255, 255, 255))
            image_buffer = []
            line_buffer = []
            depth_offset = 0
            image_location = dict()
            groups_drawn = 0
            for depth in range(len(ims)):
                screen_ims = []
                screen_scores = []
                screen_actions = []
                screen_parents = []
                screen_self = []

                for i in ims[depth]:
                    if i[4] in screen_self and remove_duplicate_states:
                        continue
                    img = i[2]
                    screen_scores.append(i[1])
                    screen_actions.append(i[0])
                    screen_parents.append(i[3])
                    screen_self.append(i[4])
                    img_surface = numpy_array_to_surface(img)

                    # Scale up the image surface
                    scaled_img_surface = pygame.transform.scale(img_surface, (image_width, image_height))
                    screen_ims.append(scaled_img_surface)

                # Blit the scaled image onto the screen
                grouped_images = dict()
                # find images of same parent
                for i in range(len(screen_ims)):
                    if screen_parents[i] in grouped_images:
                        grouped_images[screen_parents[i]].append((screen_ims[i], screen_scores[i], screen_actions[i], screen_parents[i], screen_self[i]))
                    else:
                        grouped_images[screen_parents[i]]= [(screen_ims[i], screen_scores[i], screen_actions[i], screen_parents[i], screen_self[i]), ]

                number_images_drawn = 0
                for group_id, group in enumerate(grouped_images):
                    if split_depths:
                        #line_color = (*boardState._get_unique_color((depth+depth_offset+1)/unique_parents * 255), line_opacity)
                        line_color = (*boardState._get_unique_color((group_id)/len(grouped_images) * 255), line_opacity)
                        pass
                    else:
                        #line_color = (*boardState._get_unique_color((groups_drawn+1)/unique_parents * 255), line_opacity)
                        line_color = (*boardState._get_unique_color((number_images_drawn+1)/len(screen_ims) * 255), line_opacity)
                        groups_drawn += 1

                    print("Depth: ", depth, ", Images: ", len(grouped_images[group]))
                    for i, data in enumerate(grouped_images[group]):
                        #if split_depths:
                        #    line_color = (*boardState._get_unique_color((i+1)/len(grouped_images[group]) * 255), line_opacity)
                        #else:
                        #    line_color = (*boardState._get_unique_color((number_images_drawn+1)/len(screen_ims) * 255), line_opacity)

                        im, score, actions, parent, slf = data
                        score_surface = my_font.render("Score: " + str(round(score,2)), True, snake_background_color)
                        depth_surface = my_font.render(f'Depth: {depth}', True, snake_background_color)
                        parent_surface = my_font.render(f'Parent: {round(parent,7)}', True, snake_background_color) 
                        id_surface = my_font.render(f'ID: {round(slf,7)}', True, snake_background_color) 
                        action_surfaces = []
                        for snake_id in actions:
                            _s, _ = boardState.get_snake(snake_id)
                            color_name = get_colour_name(boardState._get_unique_color(_s["m"]))
                            action_txt = color_name + ": " + actions[snake_id].name
                            _surface = my_font.render(action_txt, True, snake_background_color)
                            action_surfaces.append(_surface)

                        # centering + image offset between images
                        if split_depths:
                            x_diff = (large_width/2 - len(grouped_images[group]) * (image_width/2 + image_x_separation/2)) + i*image_width + (i+0.5)*image_x_separation
                        else:
                            x_diff = (large_width/2 - len(screen_ims) * (image_width/2 + image_x_separation/2)) + number_images_drawn*image_width + (number_images_drawn+0.5)*image_x_separation
                        
                        number_images_drawn += 1
                        # depth offset + image offset at depth
                        im_y_diff = (depth + depth_offset)*image_height + (depth + depth_offset)*image_y_separation+starting_y
                        action_y_diffs = []
                        for j in range(len(action_surfaces)):
                            action_y_diffs.append(im_y_diff-text_y_separation*(j+1))
                        score_y_diff = action_y_diffs[-1] - text_y_separation if len(action_y_diffs) > 0 else im_y_diff - text_y_separation
                        depth_y_diff = score_y_diff-text_y_separation
                        parent_y_diff = depth_y_diff-text_y_separation
                        id_y_diff = parent_y_diff-text_y_separation

                        image_location[slf] = (x_diff + image_width/2, im_y_diff + image_height/2)

                        # Need to draw images above lines
                        image_buffer.append((im, (x_diff, im_y_diff)))
                        image_buffer.append((score_surface, (x_diff, score_y_diff)))
                        for j in range(len(action_surfaces)):
                            image_buffer.append((action_surfaces[j], (x_diff, action_y_diffs[j])))
                        image_buffer.append((depth_surface, (x_diff, depth_y_diff)))
                        image_buffer.append((parent_surface, (x_diff, parent_y_diff)))
                        image_buffer.append((id_surface, (x_diff, id_y_diff)))

                        if parent >= 0: # Dont draw a line for the top node
                            start_pos = (x_diff + image_width/2, im_y_diff)
                            line_buffer.append(((start_pos, image_location[parent]), line_color))

                    if (split_depths): # and len(grouped_images) > 1): # this makes it so that only nodes with multiple parents get split # and (group_id+1) < len(grouped_images)): # This removes the gap between depths
                        depth_offset += 1

            # Update the display
            for line, color in line_buffer:
                #pygame.draw.line(line_surface, line_color, start_pos, image_location[parent], 3)
                pygame.draw.lines(large_surface, color, False, line, 3)
            
            #large_surface.blit(line_surface, (0,0))

            for im, pos in image_buffer:
                large_surface.blit(im, pos)

            print("Total images: ", total_number_of_images, ", Pruned: ", pruned_branches, ", Images drawn: ", number_images_drawn)

            drawn = True

        # Handle panning with mouse
        if is_panning:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            view_x += mouse_x - pan_start[0]
            view_y += mouse_y - pan_start[1]
            pan_start = (mouse_x, mouse_y)

        # Ensure the view doesn't go out of bounds
        view_x = max(-(large_width * zoom_scale - screen_width), min(0, view_x))
        view_y = max(-(large_height * zoom_scale - screen_height), min(0, view_y))

        # Clear the screen
        if invert_colors:
            screen.fill((0, 0, 0))
        else:
            screen.fill((255, 255, 255))

        # Render a scaled portion of the larger surface onto the screen based on the current view and scale factor
        if last_zoom != zoom_scale:
            last_zoom = zoom_scale
            scaled_width = int(large_width * zoom_scale)
            scaled_height = int(large_height * zoom_scale)
            scaled_image = pygame.transform.scale(large_surface, (scaled_width, scaled_height))
            screen.blit(scaled_image, (view_x, view_y))
        else:
            screen.blit(scaled_image, (view_x, view_y))

        pygame.display.flip()

        # Cap the frame rate to fps
        clock.tick(fps)

    pygame.quit()