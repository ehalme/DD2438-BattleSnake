# Battlesnake Python Project

## Technologies Used

This project uses [Python 3](https://www.python.org/) and [Flask](https://flask.palletsprojects.com/).

## Mini-max
### Heuristic
* Death -inf
* Run out of health -inf
* Kill friendly -
* Kill opponent +
* Get food + (depending on how low our health is)
* Hazards - (depending on health)
* Head to head collisions +/-

We will include a heatmap of "good" tiles to be in which looks at the following properties
*  distance to food
*  distance to enemies
*  distance to friendly

We will start off by searching individually, we will try to explore if we can make the search for both agents at the same time


## Notes
On start game, spawn a process for each snake so we can do parallel computing