import numpy as np
import matplotlib.pyplot as plt 

from board import Board, BoardState
from main import Action

if __name__ == "__main__":   
    s1 = {
            "id": "totally-unique-snake-id1",
            "m": 0,
            "name": "Sneky McSnek Face",
            "health": 54,
            "body": [
                {"x": 0, "y": 0},
                {"x": 1, "y": 0},
                {"x": 2, "y": 0}
            ],
            "latency": "123",
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
            "name": "Sneky McSnek Face",
            "health": 54,
            "body": [
                {"x": 5, "y": 0},
                {"x": 6, "y": 0},
                {"x": 7, "y": 0}
            ],
            "latency": "123",
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
            "name": "Sneky McSnek Face",
            "m": 2,
            "health": 54,
            "body": [
                {"x": 5, "y": 1},
                {"x": 6, "y": 1},
                {"x": 6, "y": 2}
            ],
            "latency": "123",
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

    b_example = {
                "height": 11,
                "width": 13,
                "food": [
                    {"x": 5, "y": 5},
                    {"x": 9, "y": 0},
                    {"x": 12, "y": 10}
                ],
                "hazards": [
                    {"x": 0, "y": 1},
                    {"x": 0, "y": 2}
                ],
                "snakes": [
                    s1, s2, s3
                ]
            }

    def show_boards(datas, rows, cols):
        fig = plt.figure()
        for i in range(1,len(datas)+1):
            fig.add_subplot(rows, cols, i)
            plt.imshow(datas[i-1], origin="lower", interpolation="None")
            plt.xticks(range(boardState.width))
            plt.yticks(range(boardState.height))
            plt.grid()

        plt.show()

    def get_board_img(boardState):
        board = boardState.get_board_matrix()
        data = np.zeros((boardState.width, boardState.height, 3), dtype=np.float32)

        for x in range(boardState.width):
            for y in range(boardState.height):
                if BoardState.snake_body in board[x][y]:
                    data[x,y,board[x][y][BoardState.snake_body]["m"]] = 0.8
                elif BoardState.snake_head in board[x][y]:
                    data[x,y,board[x][y][BoardState.snake_head]["m"]] = 1
                elif BoardState.food in board[x][y]:
                    data[x,y,1] = 1
                    data[x,y,0] = 1
                elif BoardState.hazard in board[x][y]:
                    data[x,y,0] = 0.5
                    data[x,y,2] = 0.5

        # Transpose the matrix
        transposed_matrix = np.transpose(data, axes=(1, 0, 2))

        return transposed_matrix


    boardState = Board(b_example, 100, -2)
    b0 = get_board_img(boardState)

    moves = {
        "totally-unique-snake-id1": Action.up, # red
        "totally-unique-snake-id2": Action.left, # green
        "totally-unique-snake-id3": Action.up # blue
    }
    boardState.move_snakes(moves)

    b1 = get_board_img(boardState)

    for snake in boardState.snakes:
        print(snake)

    show_boards((b0, b1), 1, 2)

