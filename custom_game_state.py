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
        "name": "Sneky McSnek Face2",
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