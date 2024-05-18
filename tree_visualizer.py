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

class Image:
    def __init__(self, image, depth, score, actions, parent, slf):
        self.image = image
        self.depth = depth
        self.score = score
        self.actions = actions
        self.parent = parent
        self.slf = slf

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

pruned_branches = 0
img_ids = 1

def numpy_array_to_surface(array):
    # Flip the array along the y-axis before creating the surface
    flipped_array = np.flipud(array)
    return pygame.surfarray.make_surface(flipped_array.swapaxes(0, 1))

def start_minimax(game_state: typing.Dict, heuristic: Heuristic, max_depth: int, max_health: int, hazard_decay: int, step_decay: int) -> typing.Tuple[Action, typing.Dict[int, typing.List[Image]]]:
    """
    Does a minimax run from this starting game_state using the
    specified heuristic.
    Return an Action
    """
    global img_ids

    timeout = game_state["game"]["timeout"]
    latency = int(game_state["you"]["latency"]) if game_state["you"]["latency"] != '' else 100
    calculation_time = timeout - latency - 60 # X ms for padding
    calculation_time += 500000000
    calculation_time *= 1e6 # convert ms (10^3) to ns (10^9)
    
    food_spawn_chances = np.random.rand(20) # Create a vector of random numbers so that every environment spawns food in the same way
    my_snake = game_state['you']['id']
    init_board = Board(game_state["board"], max_health=max_health, hazard_decay=hazard_decay, step_decay=step_decay, food_spawn_chances=food_spawn_chances)

    start_time = time.time_ns()

    friendly_snakes, enemy_snakes, my_snake_alive = get_other_snakes(init_board, my_snake)

    all_friendly = friendly_snakes + [my_snake]
    action_combos = get_children(init_board, all_friendly)

    transposition_table = dict()
    highest_score = -math.inf
    best_actions = None
    all_images = dict()

    #print(my_snake)
    for depth in range(1,max_depth+1):
        images = list()

        for i, actions in enumerate(action_combos):
            new_board = init_board.copy()
    
            new_board.move_snakes(actions)

            slf = img_ids
            img_ids += 1

            score = minimax(new_board, my_snake, heuristic, calculation_time, start_time, depth, False, -math.inf, math.inf, transposition_table, slf, images)
    
            #print("Score: ", score, ", Actions: ", actions)

            images.append(Image(new_board.get_board_img(), depth+1, score, actions, 0, slf))
            
            if score > highest_score:
                highest_score = score
                best_actions = actions
            
        all_images[depth] = images

        used_time = time.time_ns() - start_time
        if used_time >= calculation_time:
            break

    print("Minimax time: ", (time.time_ns() - start_time) * 1e-6)

    if best_actions is None:
        return Action.up
    
    return best_actions[my_snake], all_images

    # start the minimax alg from init_board
    # make copies of the board for width and friendly/enemy snakes lists
    # used Board.move_snakes() when going depth
    # use heuristic.get_score(board) to get the score of the current game state

def minimax(board: Board, my_snake: str, heuristic: Heuristic, calculation_time: int, start_time: int, depth: int, maximizing: bool, alpha: int, beta: int, transposition_table: typing.Dict, parent: int, images: list) -> float: # TODO: edit object
    global pruned_branches, img_ids
    observation = board.get_sparse_observation()
    if observation in transposition_table:
        return transposition_table[observation]
    
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

            slf = img_ids #1 + parent + (i+1)*0.00001
            img_ids += 1

            new_board.move_snakes(actions)
            score = max(score, minimax(new_board, my_snake, heuristic, calculation_time, start_time, depth - 1, False, alpha, beta, transposition_table, slf, images))
            alpha = max(alpha, score)

            images.append(Image(new_board.get_board_img(), depth, score, actions, parent, slf))

            if beta <= alpha and not disable_pruning:
                transposition_table[observation] = score
                pruned_branches += 1
                break
    else:
        score = math.inf

        action_combos = get_children(board, enemy_snakes)

        for i, actions in enumerate(action_combos):
            new_board = board.copy()

            slf = img_ids #1 + parent + (i+1)*0.001
            img_ids += 1

            new_board.move_snakes(actions)
            score = min(score, minimax(new_board, my_snake, heuristic, calculation_time, start_time, depth - 1, True, alpha, beta, transposition_table, slf, images))
            beta = min(beta, score)

            images.append(Image(new_board.get_board_img(), depth, score, actions, parent, slf))

            if beta <= alpha and not disable_pruning:
                transposition_table[observation] = score
                pruned_branches += 1
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
    max_depth = 4
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
    # Display which graph is drawn on, increase for larger depths (max ~50_000). Zooming in is slow on large images
    large_width = 15_000
    large_height = 15_000
    # Set seed
    #random.seed(0)
    # Custom game state
    use_predefined_game_state = True
    # iterative deepening layer (max of max_depth, min of 1)
    depth_layer = max_depth
    # disable pruning
    disable_pruning = False

    ####################################################################################################################
    snake_background_color = (0,0,0)

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

    action, all_images = start_minimax(game_state, heuristic, max_depth=max_depth, max_health=100, hazard_decay=0, step_decay=1)
    print("Best action: ", action.name)

    t = []
    for depth in all_images:
        for img in all_images[depth]:
            img.depth = depth - img.depth + 2
            if img.depth not in t:
                t.append(img.depth)

        all_images[depth].insert(0, Image(initial_node_im, 0, 0, {}, -1, 0))

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
            large_surface.fill((255, 255, 255))
            depth_offset = 0
            image_location = dict()
            total_number_of_images = 0
            image_buffer = []

            #for depth in all_images:
            img_draw_in_depth = dict()
            imgs_per_depth = dict()
            image_location = dict()
            children_per_parent = dict()
            # set up dicts
            for img in all_images[depth_layer]:
                if img.depth not in imgs_per_depth:
                    imgs_per_depth[img.depth] = 0

                if img.parent not in children_per_parent:
                    children_per_parent[img.parent] = 0

                children_per_parent[img.parent] += 1
                imgs_per_depth[img.depth] += 1
                img_draw_in_depth[img.depth] = 0

            for img in all_images[depth_layer]:
                img_surface = numpy_array_to_surface(img.image)

                # Scale up the image surface
                scaled_img_surface = pygame.transform.scale(img_surface, (image_width, image_height))

                score_surface = my_font.render("Score: " + str(round(img.score,2)), True, snake_background_color)
                depth_surface = my_font.render(f'Depth: {img.depth}', True, snake_background_color)
                parent_surface = my_font.render(f'Parent: {round(img.parent,7)}', True, snake_background_color) 
                id_surface = my_font.render(f'ID: {round(img.slf,7)}', True, snake_background_color) 
                action_surfaces = []
                for snake_id in img.actions:
                    _s, _ = boardState.get_snake(snake_id)
                    color_name = get_colour_name(boardState._get_unique_color(_s["m"]))
                    action_txt = color_name + ": " + img.actions[snake_id].name
                    _surface = my_font.render(action_txt, True, snake_background_color)
                    action_surfaces.append(_surface)

                x_diff = (large_width/2 - imgs_per_depth[img.depth] * (image_width/2 + image_x_separation/2)) + img_draw_in_depth[img.depth]*image_width + (img_draw_in_depth[img.depth]+0.5)*image_x_separation
                im_y_diff = (img.depth + depth_offset)*image_height + (img.depth + depth_offset)*image_y_separation+starting_y
                action_y_diffs = []
                for j in range(len(action_surfaces)):
                    action_y_diffs.append(im_y_diff-text_y_separation*(j+1))
                score_y_diff = action_y_diffs[-1] - text_y_separation if len(action_y_diffs) > 0 else im_y_diff - text_y_separation
                depth_y_diff = score_y_diff-text_y_separation
                parent_y_diff = depth_y_diff-text_y_separation
                id_y_diff = parent_y_diff-text_y_separation

                image_buffer.append((scaled_img_surface, (x_diff, im_y_diff)))
                image_buffer.append((score_surface, (x_diff, score_y_diff)))
                for j in range(len(action_surfaces)):
                    image_buffer.append((action_surfaces[j], (x_diff, action_y_diffs[j])))
                image_buffer.append((depth_surface, (x_diff, depth_y_diff)))
                image_buffer.append((parent_surface, (x_diff, parent_y_diff)))
                image_buffer.append((id_surface, (x_diff, id_y_diff)))

                image_location[img.slf] = (x_diff + image_width/2, im_y_diff + image_height/2)
                img_draw_in_depth[img.depth] += 1
                total_number_of_images += 1

            parents_drawn = dict()
            color_idx = dict()
            for d in children_per_parent:
                parents_drawn[d] = 0

            for d in img_draw_in_depth:
                color_idx[d] = 0

            # lines
            for img in all_images[depth_layer]:
                if img.parent != -1: # Dont draw a line for the top node
                    start_pos = list(image_location[img.slf])
                    start_pos[1] -= image_height/2
                    end_pos = list(image_location[img.parent])
                    end_pos[1] += image_height/2
                    num_parents = imgs_per_depth[img.depth-1]
                    line_color = (*boardState._get_unique_color((color_idx[img.depth]+1)/num_parents * 255), line_opacity)

                    parents_drawn[img.parent] += 1

                    if parents_drawn[img.parent] == children_per_parent[img.parent]:
                        color_idx[img.depth] += 1

                    pygame.draw.line(large_surface, line_color, start_pos, end_pos, 3)

            for img, pos in image_buffer:
                large_surface.blit(img, pos)
            

            print("Total images: ", total_number_of_images, ", Pruned: ", pruned_branches)

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