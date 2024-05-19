from board import Board
from minimax import start_minimax
from heuristics import Heuristic
from multiprocessing.pool import Pool
import wandb
from weights import weights
import numpy as np


def train():
    """Team 0 and 1, team 0 optimizing"""
    run = wandb.init()

    optimizing_weights = {
        "food_distance": wandb.config.food_distance,
        "eat_food": wandb.config.eat_food,
        "death": -1000,
        "enemy_killed": 100,
        "friendly_killed": -100,
        "health":  wandb.config.health,
        "flood_fill":  wandb.config.flood_fill,
        "length":  wandb.config.length
    }
    
    mean_steps = []
    for iteration in range(10):
        max_depth = 30
        max_steps = 500
        
        boardState = Board(None, max_health=100, hazard_decay=0, step_decay=1, move_all_snakes=True, print_logs=False)

        # Get teams
        teams = dict()
        for snake in boardState.snakes:
            if snake["name"] not in teams:
                teams[snake["name"]] = 1

        teams[list(teams.keys())[0]] = 0

        optimizing_snakes = [x for x in boardState.snakes if teams[x["name"]] == 0]

        # Main loop
        running = True 
        steps = 0
        while running:
            actions = dict()

            for snake in boardState.snakes:
                game_state = {
                    "game": {"timeout": 500, },
                    "board": boardState.get_json_board(),
                    "you": snake,
                }

                if teams[snake["name"]] == 0:
                    heuristic = Heuristic(optimizing_weights)
                else:
                    heuristic = Heuristic(weights)

                action = start_minimax(game_state, heuristic, max_depth=max_depth, max_health=100, hazard_decay=0, step_decay=1, print_logs=False)
                actions[snake["id"]] = action

            boardState.move_snakes(actions)
            steps += 1

            team_alive = 0
            opponent_alive = 0
            for snake in boardState.snakes:            
                if teams[snake["name"]] == 0:
                    team_alive += 1
                else:
                    opponent_alive += 1

            if team_alive == 0 or opponent_alive == 0 or steps > max_steps:
                running = False

        mean_steps.append(steps)

        win_reason = -1
        if team_alive == 0:
            win_reason = 0
        elif opponent_alive == 0:
            win_reason = 1
        elif steps > max_steps:
            win_reason = 2

        logs = {"steps": steps, "mean_steps": np.mean(mean_steps), "win reason": win_reason}

        for i, _snake in enumerate(optimizing_snakes):
            snake, is_alive = boardState.get_snake(_snake["id"])
            
            logs[f"length {i}"] = snake["length"]
            logs[f"alive {i}"] = int(is_alive)
        
        wandb.log(logs)

def main(sweep_id, i):
    print("Running: ", i)

    wandb.agent(sweep_id=sweep_id, function=train, count=200)

if __name__ == "__main__":
    wandb.login()

    # Sweep config
    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep1',
        'metric': {'goal': 'maximize', 'name': 'mean_steps'},
        'parameters': 
        {
            'food_distance': {'max': 1, 'min': 0, 'distribution': 'uniform'},
            'eat_food': {'max': 1, 'min': 0, 'distribution': 'uniform'},
            'health': {'max': 1, 'min': 0, 'distribution': 'uniform'},
            'flood_fill': {'max': 1, 'min': 0, 'distribution': 'uniform'},
            'length': {'max': 1, 'min': 0, 'distribution': 'uniform'},
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='battlesnakes'
    )

    #sweep_id = "lni/battlesnakes/gok81k90"
    
    cores = 20
    with Pool(cores) as p:
        p.starmap(main, zip([sweep_id,]*cores, range(cores)))