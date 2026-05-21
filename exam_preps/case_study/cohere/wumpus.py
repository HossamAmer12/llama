import random
import cohere

# Dodecahedron cave: 20 rooms, each connected to exactly 3 others
CAVE = {
    1:  [2, 5, 8],
    2:  [1, 3, 10],
    3:  [2, 4, 12],
    4:  [3, 5, 14],
    5:  [1, 4, 6],
    6:  [5, 7, 15],
    7:  [6, 8, 17],
    8:  [1, 7, 9],
    9:  [8, 10, 18],
    10: [2, 9, 11],
    11: [10, 12, 19],
    12: [3, 11, 13],
    13: [12, 14, 20],
    14: [4, 13, 15],
    15: [6, 14, 16],
    16: [15, 17, 20],
    17: [7, 16, 18],
    18: [9, 17, 19],
    19: [11, 18, 20],
    20: [13, 16, 19],
}


# 1) Define the environment: 20 rooms, each room has a list of adjacent rooms
class WumpusEnvironment:
    def __init__(self):
        self.cave = CAVE
        self._place_entities()

    def _place_entities(self):
        rooms = list(self.cave.keys())
        locations = random.sample(rooms, 6)  # ensures all distinct rooms
        self.player = locations[0]
        self.wumpus = locations[1]
        self.bats = [locations[2], locations[3]]
        self.pits = [locations[4], locations[5]]

    def neighbors(self, room):
        return self.cave[room]

    def __repr__(self):
        return (
            f"Player: room {self.player}\n"
            f"Wumpus: room {self.wumpus}\n"
            f"Bats:   rooms {self.bats}\n"
            f"Pits:   rooms {self.pits}\n"
        )


# 2) Define the player: has a position (current room), number of arrows, and percepts (what the player can sense in the current room)
class Player:
    def __init__(self, room):
        self.room = room
        self.arrows = 5

    def __repr__(self):
        return f"Player(room={self.room}, arrows={self.arrows})"

    def shoot_arrow(self, target_room):
        if self.arrows <= 0:
            return -1 # No arrows left
        self.arrows -= 1
        return target_room

# 3) Define a function that does the LLM call: given the player's current position and percepts, return the action the player should take: "Kill the wompus", "Go to this room", etc.
SYSTEM_PROMPT = """You are an intelligent agent playing Hunt the Wumpus. Your goal is to kill the Wumpus and survive.

RULES:
- If you enter a room containing the Wumpus, you are eaten — game over.
- If you enter a room with a bottomless pit, you fall in — game over.
- If you enter a room with a giant bat, it transports you to a random empty room.
- You may shoot an arrow into any adjacent room. If the Wumpus is there, you win. Each shot costs one arrow.
- You lose if you run out of arrows without killing the Wumpus.

STRATEGY:
- Strongly favour moves and shots that lead to winning.
- Avoid rooms likely to contain a pit or the Wumpus unless shooting.
- Only shoot when you have strong reason to believe the Wumpus is in the target room.
- Conserve arrows — you only have a few.

RESPONSE FORMAT:
Think step by step about the situation, then end with exactly one action line:
  MOVE <room_number>   — move into an adjacent room
  SHOOT <room_number>  — shoot an arrow into an adjacent room"""


def get_llm_action(env: WumpusEnvironment, player: Player, arrow_history: list) -> str:
    return "test"
    # co = cohere.ClientV2()

    # neighbors = env.neighbors(player.room)
    # user_message = (
    #     f"Current room: {player.room}\n"
    #     f"Adjacent rooms: {neighbors}\n"
    #     f"Arrows remaining: {player.arrows}\n"
    #     f"Arrow count history (most recent last): {arrow_history}\n\n"
    #     "Think step by step, then give your action."
    # )

    # response = co.chat(
    #     model="command-a-03-2025",
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user",   "content": user_message},
    #     ],
    # )

    # return response.message.content[0].text


if __name__ == "__main__":
    env = WumpusEnvironment()
    player = Player(env.player)
    arrow_history = [player.arrows]

    print(env)
    print(player)
    print(f"Neighbors of player's room {player.room}: {env.neighbors(player.room)}")
    print("\n--- LLM Decision ---")
    action = get_llm_action(env, player, arrow_history)
    print(action)



# 3) Define the game loop: the player starts in a random empty room, senses the environment, decides on an action


# 4) Define a function that does the LLM call: given the player's current position and percepts, return the action the player should take: "Kill the wompus", "Go to this room", etc.















