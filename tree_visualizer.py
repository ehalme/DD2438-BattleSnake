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
    calculation_time *= 1e6 # convert ms (10^3) to ns (10^9)
    
    my_snake = game_state['you']['id']
    init_board = Board(game_state["board"], max_health=max_health, hazard_decay=hazard_decay, step_decay=step_decay)

    start_time = time.time_ns()

    friendly_snakes, enemy_snakes, my_snake_alive = get_other_snakes(init_board, my_snake)

    all_friendly = friendly_snakes + [my_snake]
    action_combos = get_children(init_board, all_friendly)

    highest_score = -math.inf
    best_actions = None

    for i, actions in enumerate(action_combos):
        new_board = init_board.copy()

        new_board.move_snakes(actions)
    
        score = minimax(new_board, my_snake, heuristic, calculation_time, start_time, max_depth-1, False, 1+0.1*(i+1))

        #print("Score: ", score, ", Actions: ", actions)
        
        if (max_depth) in images:
            images[max_depth].append((actions, score, new_board.get_board_img(), 0, 1+0.1*(i+1)))
        else:
            images[max_depth] = [(actions, score, new_board.get_board_img(), 0, 1+0.1*(i+1)),]

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

def minimax(board: Board, my_snake: str, heuristic: Heuristic, calculation_time: int, start_time: int, depth: int, maximizing: bool, parent: int) -> float: # TODO: edit object
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
            score = max(score, minimax(new_board, my_snake, heuristic, calculation_time, start_time, depth - 1, False, parent+1+0.001*(i+1)))
            if (depth) in images:
                images[depth].append((actions, score, new_board.get_board_img(), parent, parent+1+0.001*(i+1)))
            else:
                images[depth] = [(actions, score, new_board.get_board_img(), parent, parent+1+0.001*(i+1)),]
        
        return score
        
    else:
        score = math.inf

        action_combos = get_children(board, enemy_snakes)

        for i, actions in enumerate(action_combos):
            new_board = board.copy()

            new_board.move_snakes(actions)
            score = min(score, minimax(new_board, my_snake, heuristic, calculation_time, start_time, depth - 1, True, parent+1+0.001*(i+1)))
            if (depth) in images:
                images[depth].append((actions, score, new_board.get_board_img(), parent, parent+1+0.001*(i+1)))
            else:
                images[depth] = [(actions, score, new_board.get_board_img(), parent, parent+1+0.001*(i+1)),]
        
        return score

if __name__ == "__main__":   
    s1 = {
            "id": "id0",
            "m": 50,
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
            "id": "id1",
            "m": 120,
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
            "id": "id2",
            "name": "Sneky McSnek Face2",
            "m": 160,
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
            "id": "id3",
            "name": "Sneky McSnek Face3",
            "m": 200,
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
    
    manual_control = False
    max_depth = 2

    manual_snake = s1
    heuristic = Heuristic(weights)
    
    boardState = Board(b_example, max_health=100, hazard_decay=2, step_decay=1, print_logs=True) # temp
    initial_node_im = boardState.get_board_img()
  
    # Scale factor for rendering the image larger
    SCALE_FACTOR = 10

    # Initialize Pygame
    pygame.init()
    image_width = b_example["width"] * SCALE_FACTOR
    image_height = b_example["height"] * SCALE_FACTOR
    screen_width = 1500
    screen_height = 1000
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Image Rendering")
    pygame.font.init()
    my_font = pygame.font.SysFont('Comic Sans MS', 10)

    action = start_minimax(game_state, heuristic, max_depth=max_depth, max_health=100, hazard_decay=0, step_decay=1)
    print("Best action: ", action.name)

    # images = (actions, score, new_board.get_board_img(), parent, self)
    ims = [[] for i in range(len(images))]
    for depth in images:
        for img in images[depth]:
            ims[(max_depth) - depth].append(img)

    ims.insert(0,[([], 0, initial_node_im, -1, 0),])

    unique_parents = set()
    for depth in ims:
        for data in depth:
            unique_parents.add(data[3])

    unique_parents = len(unique_parents)

    # Main loop
    running = True
    drawn = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False            

        if not drawn:
            screen.fill((255,255,255))
            image_buffer = []
            depth_offset = 0
            image_seperation = 40
            image_location = dict()
            start_color = pygame.Color(255,0,0,100)
            end_color = pygame.Color(0,255,255,100)
            for depth in range(len(ims)):
                screen_ims = []
                screen_scores = []
                screen_actions = []
                screen_parents = []
                screen_self = []

                for i in ims[depth]:
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

                for group in grouped_images:
                    line_color = pygame.Color.lerp(start_color, end_color, (depth+depth_offset)/unique_parents)
                    print("Depth: ", depth, ", Images: ", len(grouped_images[group]))
                    for i, data in enumerate(grouped_images[group]):
                        im, score, actions, parent, slf = data
                        score_surface = my_font.render("Score: " + str(round(score,2)), True, (0, 0, 0))
                        data_surface = my_font.render(f'Depth: {depth}, Parent: {round(parent,5)}, ID: {round(slf,5)}', True, (0,0,0))
                        action_txt = ""
                        for snake_id in actions:
                            _s, _ = boardState.get_snake(snake_id)
                            color_name = get_colour_name(boardState._get_unique_color(_s["m"]))
                            action_txt += color_name + ": " + actions[snake_id].name + ", "
                        action_surface = my_font.render(action_txt, True, (0, 0, 0))

                        # centering + image offset between images
                        x_diff = (screen_width/2 - len(grouped_images[group]) * (image_width/2 + image_seperation/2)) + i*image_width + (i+0.5)*image_seperation
                        # depth offset + image offset at depth
                        im_y_diff = (depth + depth_offset)*image_height + (depth + depth_offset)*80+65
                        action_y_diff = (depth + depth_offset)*image_height + (depth + depth_offset)*80+15
                        score_y_diff = (depth + depth_offset)*image_height + (depth + depth_offset)*80+30
                        data_y_diff = (depth + depth_offset)*image_height + (depth + depth_offset)*80+45

                        image_location[slf] = (x_diff + image_width/2, im_y_diff + image_height/2)

                        # Need to draw images above lines
                        image_buffer.append((im, (x_diff, im_y_diff)))
                        image_buffer.append((score_surface, (x_diff, score_y_diff)))
                        image_buffer.append((action_surface, (x_diff, action_y_diff)))
                        image_buffer.append((data_surface, (x_diff, data_y_diff)))

                        if parent >= 0:
                            line_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                            pygame.draw.line(line_surf, line_color, image_location[slf], image_location[parent], 3)
                            screen.blit(line_surf, (0,0))

                    if (len(grouped_images) > 1):
                            depth_offset += 1

            # Update the display
            for im, pos in image_buffer:
                screen.blit(im, pos)

            pygame.display.flip()
            drawn = True

    pygame.quit()