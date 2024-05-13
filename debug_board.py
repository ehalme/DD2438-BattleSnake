import matplotlib.pyplot as plt 

from board import Board, Action

def show_boards(datas, rows, cols):
    fig = plt.figure()
    for i in range(1,len(datas)+1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(datas[i-1], origin="lower", interpolation="None")
        plt.xticks(range(boardState.width))
        plt.yticks(range(boardState.height))
        plt.grid()

    plt.show()

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


    boardState = Board(b_example, max_health=100, hazard_decay=2, step_decay=1)
    b0 = boardState.get_board_img

    moves = {
        "totally-unique-snake-id1": Action.up, # red
        "totally-unique-snake-id2": Action.left, # green
        "totally-unique-snake-id3": Action.up # blue
    }
    moves2 = {
        "totally-unique-snake-id1": Action.right, # red
        "totally-unique-snake-id2": Action.up, # green
        "totally-unique-snake-id3": Action.right # blue
    }

    boardState.move_snakes(moves)
    b1 = boardState.get_board_img

    print(1)
    for snake in boardState.snakes:
        print(snake)

    boardState.move_snakes(moves)
    b2 = boardState.get_board_img

    print(2)
    for snake in boardState.snakes:
        print(snake)

    boardState.move_snakes(moves)
    b3 = boardState.get_board_img

    print(3)
    for snake in boardState.snakes:
        print(snake)

    boardState.move_snakes(moves)
    b4 = boardState.get_board_img

    print(4)
    for snake in boardState.snakes:
        print(snake)

    boardState.move_snakes(moves2)
    for i in range(2):
        boardState.move_snakes(moves)
    b5 = boardState.get_board_img

    print(5)
    for snake in boardState.snakes:
        print(snake)

    show_boards((b0, b1, b2, b3, b4, b5), 2, 3)

