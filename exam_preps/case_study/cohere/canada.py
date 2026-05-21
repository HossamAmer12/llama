# Graph:
#   START → generate → execute ──success──► evaluate ──pass──► END
#                        ↑                               │
#                        └──────── error / fail ─────────┘

from ast import If, Return


Proof-of-concepts:
    
(1) API: LLM (room, adjacent rooms) -> 
Nearby Wumpus: "You smell something terrible nearby."
Nearby bat: "You hear a rustling."
Nearby pit: "You feel a cold wind blowing from a nearby cavern."

action the player should take: "Kill the wompus"
"Go to this room"

API..... Player position, Rooms, get back information
Maintain all the context...Symbolic llm... 
Return sensing output.... when the player makes a move... 



(2) Player Construct: Player class
(i) 5 arrows -> how many arrows left
(ii) Room number -> where the player is
(iii) Percepts -> what the player can sense in the current room (e.g., smell, sound, wind)

(3) Game Loop:
    
Simple design --- Edge cases (I am sure.. we need to discuss this more)
Generate room --> 
Room0 [1, 2, 4]
Room1 [4, 2, 3]
Romm2 []
.
..
Room20 []

In the cave there are:

One Wumpus --> Random + Empty room
Two  bats --> Random + Empty room
Two pits ---> Random + Empty room


(4) Agent/Player loop

Start (random empty room)
--> Sense (percepts)
--> Decide (action)
--> Act (move to another room, shoot an arrow, etc.)
--> Update (player state, game state)
--> Repeat until win/lose condition is met


Win:
- Kill the Wumpus (shoot an arrow into the room where the Wumpus is located)

Lose:
    Wompus + Player
    Pit + Player
    Bat + Player 
    














# device placement
# collator

# 1. Generation

# --> Align with the format
# --> 


# 2. Execution

# --> Timeouts
# ---> Failures (retries)

# Import sandbox errors 
# Stuck Hypothesis


# 3. Memory -- short-term and long-term history
# Retry.. and what it failed


# 4. Evaluation

# Invocation accuracy measures whether the LLM correctly decided to call a tool when it should, or avoided calling one when it should not. For example, the model may call a tool when it should answer the question directly.

# Selection accuracy measures whether the LLM called the correct tools, usually by keeping track of a ground truth trajectory that includes a list of necessary tools for solving a particular problem.

# Structural accuracy and schema validity measure whether the structure of a tool call is correct. For example, our model should not include wrong arguments in a tool call or provide an incorrect call structure.

# Trajectory accuracy looks at the sequence of tool calls made by the model when solving a problem and compares them to ground truth in some way (e.g., correct call order, correct selection, using unnecessary tools, and more).

# Pass@1

# Memory... context



