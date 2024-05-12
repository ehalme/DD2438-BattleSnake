# Battlesnake Python Project

## Technologies Used

This project uses [Python 3](https://www.python.org/) and [Flask](https://flask.palletsprojects.com/).

## Mini-max
### Heuristic
We will include a heatmap of "good" tiles to be in which looks at the following properties
* Distance to food
* Distance to enemies
* Distance to friendly
* Death -inf
* Run out of health -inf
* Kill friendly -
* Kill opponent +
* Get food + (depending on how low our health is)
* Hazards - (depending on health)
* Head to head collisions +/-

We will start off by searching individually, we will try to explore if we can make the search for both agents at the same time


## Notes
On start game, spawn a process for each snake so we can do parallel computing
Can try to allow snakes to do X amount of moves before we let other snakes do actions in our tree search?