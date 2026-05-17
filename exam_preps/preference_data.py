"""
preference_data.py — 100 handcrafted math preference pairs for DPO
Each entry: {"prompt": "Q: ...\nA:", "chosen": " <correct>", "rejected": " <wrong>"}
Rejected answers use realistic mistake patterns (wrong op, off-by-one, partial, unit error)
"""

import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor


def load_preference_data():
    """Return the raw preference pairs list."""
    return PREFERENCE_DATA


def tokenize_example(example: dict, tokenizer: SentencePieceProcessor):
    """Tokenize one preference pair.

    Returns:
        input_ids_w       : (T_w,) chosen  tokens  (prompt + chosen  response)
        input_ids_l       : (T_l,) rejected tokens (prompt + rejected response)
        response_start_idx: int — index where the response begins (= prompt length)
    """
    prompt_ids   = tokenizer.encode(example["prompt"],   out_type=int, add_bos=True,  add_eos=False)
    chosen_ids   = tokenizer.encode(example["chosen"],   out_type=int, add_bos=False, add_eos=True)
    rejected_ids = tokenizer.encode(example["rejected"], out_type=int, add_bos=False, add_eos=True)

    input_ids_w = torch.tensor(prompt_ids + chosen_ids,   dtype=torch.long)
    input_ids_l = torch.tensor(prompt_ids + rejected_ids, dtype=torch.long)
    return input_ids_w, input_ids_l, len(prompt_ids)


def pad_to_same_length(sequences, pad_id: int = 0):
    """Pad a list of 1-D tensors to the length of the longest one."""
    max_len = max(s.shape[0] for s in sequences)
    padded  = [F.pad(s, (0, max_len - s.shape[0]), value=pad_id) for s in sequences]
    return torch.stack(padded)


def get_preference_batch(
    data: list,
    tokenizer: SentencePieceProcessor,
    batch_size: int,
    device: str,
):
    """Sample a random batch, tokenize on-the-fly, and pad to same length.

    Returns:
        input_ids_w        : (B, T) chosen  sequences
        input_ids_l        : (B, T) rejected sequences
        response_start_idx : int — min prompt length in batch (safe mask boundary)
    """
    batch     = [data[i] for i in torch.randint(0, len(data), (batch_size,)).tolist()]
    tokenized = [tokenize_example(ex, tokenizer) for ex in batch]
    w_seqs, l_seqs, prompt_lens = zip(*tokenized)

    return (
        pad_to_same_length(w_seqs).to(device),
        pad_to_same_length(l_seqs).to(device),
        min(prompt_lens),
    )

PREFERENCE_DATA = [

    # ── 1. Distance = speed × time (20 examples) ──────────────────────────────
    {
        "prompt":   "Q: A car travels at 60 mph for 3 hours. How far does it travel?\nA:",
        "chosen":   " 180 miles",
        "rejected": " 63 miles",        # addition instead of multiply
    },
    {
        "prompt":   "Q: A train moves at 90 mph for 2 hours. How far does it travel?\nA:",
        "chosen":   " 180 miles",
        "rejected": " 92 miles",        # addition
    },
    {
        "prompt":   "Q: A cyclist rides at 15 mph for 4 hours. How far do they travel?\nA:",
        "chosen":   " 60 miles",
        "rejected": " 19 miles",        # addition
    },
    {
        "prompt":   "Q: A plane flies at 500 mph for 6 hours. How far does it fly?\nA:",
        "chosen":   " 3000 miles",
        "rejected": " 506 miles",       # addition
    },
    {
        "prompt":   "Q: A runner jogs at 8 mph for 1.5 hours. How far do they run?\nA:",
        "chosen":   " 12 miles",
        "rejected": " 8 miles",         # ignored the 1.5, used 1 hour
    },
    {
        "prompt":   "Q: A boat travels at 20 mph for 2.5 hours. How far does it go?\nA:",
        "chosen":   " 50 miles",
        "rejected": " 40 miles",        # used 2 hours instead of 2.5
    },
    {
        "prompt":   "Q: A bus goes 45 mph for 4 hours. How many miles does it cover?\nA:",
        "chosen":   " 180 miles",
        "rejected": " 49 miles",        # addition
    },
    {
        "prompt":   "Q: A motorcycle rides at 70 mph for 3 hours. How far does it travel?\nA:",
        "chosen":   " 210 miles",
        "rejected": " 140 miles",       # used 2 hours instead of 3
    },
    {
        "prompt":   "Q: A hiker walks at 3 mph for 5 hours. How far do they walk?\nA:",
        "chosen":   " 15 miles",
        "rejected": " 8 miles",         # addition
    },
    {
        "prompt":   "Q: A truck drives at 55 mph for 2 hours. How far does it travel?\nA:",
        "chosen":   " 110 miles",
        "rejected": " 57 miles",        # addition
    },
    {
        "prompt":   "Q: A ship sails at 25 knots for 4 hours. How far does it travel?\nA:",
        "chosen":   " 100 nautical miles",
        "rejected": " 29 nautical miles",  # addition
    },
    {
        "prompt":   "Q: A car travels at 80 mph for 2.5 hours. How far does it go?\nA:",
        "chosen":   " 200 miles",
        "rejected": " 160 miles",       # used 2 hours
    },
    {
        "prompt":   "Q: A train travels at 120 mph for 1.5 hours. How far does it go?\nA:",
        "chosen":   " 180 miles",
        "rejected": " 120 miles",       # ignored the 0.5
    },
    {
        "prompt":   "Q: A cyclist goes 12 mph for 3 hours. How far do they travel?\nA:",
        "chosen":   " 36 miles",
        "rejected": " 15 miles",        # addition
    },
    {
        "prompt":   "Q: A car drives at 65 mph for 2 hours. How far does it travel?\nA:",
        "chosen":   " 130 miles",
        "rejected": " 67 miles",        # addition
    },
    {
        "prompt":   "Q: A runner goes 6 mph for 2.5 hours. How far do they run?\nA:",
        "chosen":   " 15 miles",
        "rejected": " 12 miles",        # used 2 hours
    },
    {
        "prompt":   "Q: A plane flies at 400 mph for 3.5 hours. How far does it travel?\nA:",
        "chosen":   " 1400 miles",
        "rejected": " 1200 miles",      # used 3 hours
    },
    {
        "prompt":   "Q: A bus travels at 50 mph for 1.5 hours. How far does it go?\nA:",
        "chosen":   " 75 miles",
        "rejected": " 50 miles",        # ignored 0.5
    },
    {
        "prompt":   "Q: A boat sails at 18 mph for 3 hours. How far does it travel?\nA:",
        "chosen":   " 54 miles",
        "rejected": " 21 miles",        # addition
    },
    {
        "prompt":   "Q: A hiker walks at 4 mph for 6 hours. How far do they walk?\nA:",
        "chosen":   " 24 miles",
        "rejected": " 10 miles",        # addition
    },

    # ── 2. Total cost = price × quantity (20 examples) ────────────────────────
    {
        "prompt":   "Q: Apples cost $2 each. How much do 8 apples cost?\nA:",
        "chosen":   " $16",
        "rejected": " $10",             # addition
    },
    {
        "prompt":   "Q: A pen costs $3. How much do 7 pens cost?\nA:",
        "chosen":   " $21",
        "rejected": " $10",             # addition
    },
    {
        "prompt":   "Q: A notebook costs $5. How much do 12 notebooks cost?\nA:",
        "chosen":   " $60",
        "rejected": " $17",             # addition
    },
    {
        "prompt":   "Q: A coffee costs $4.50. How much do 4 coffees cost?\nA:",
        "chosen":   " $18.00",
        "rejected": " $16.00",          # used $4 instead of $4.50
    },
    {
        "prompt":   "Q: A ticket costs $12. How much do 5 tickets cost?\nA:",
        "chosen":   " $60",
        "rejected": " $17",             # addition
    },
    {
        "prompt":   "Q: A book costs $15. How much do 6 books cost?\nA:",
        "chosen":   " $90",
        "rejected": " $21",             # addition
    },
    {
        "prompt":   "Q: A sandwich costs $6.50. How much do 3 sandwiches cost?\nA:",
        "chosen":   " $19.50",
        "rejected": " $18.00",          # used $6 instead of $6.50
    },
    {
        "prompt":   "Q: A shirt costs $25. How much do 4 shirts cost?\nA:",
        "chosen":   " $100",
        "rejected": " $29",             # addition
    },
    {
        "prompt":   "Q: A gallon of milk costs $3.50. How much do 4 gallons cost?\nA:",
        "chosen":   " $14.00",
        "rejected": " $12.00",          # used $3 instead of $3.50
    },
    {
        "prompt":   "Q: A movie ticket costs $11. How much do 3 tickets cost?\nA:",
        "chosen":   " $33",
        "rejected": " $14",             # addition
    },
    {
        "prompt":   "Q: A pizza costs $9. How much do 5 pizzas cost?\nA:",
        "chosen":   " $45",
        "rejected": " $14",             # addition
    },
    {
        "prompt":   "Q: A toy costs $7.50. How much do 6 toys cost?\nA:",
        "chosen":   " $45.00",
        "rejected": " $42.00",          # used $7 instead of $7.50
    },
    {
        "prompt":   "Q: A bus pass costs $30. How much do 4 passes cost?\nA:",
        "chosen":   " $120",
        "rejected": " $34",             # addition
    },
    {
        "prompt":   "Q: A banana costs $0.50. How much do 10 bananas cost?\nA:",
        "chosen":   " $5.00",
        "rejected": " $10.50",          # added instead of multiplied
    },
    {
        "prompt":   "Q: A candle costs $8. How much do 9 candles cost?\nA:",
        "chosen":   " $72",
        "rejected": " $17",             # addition
    },
    {
        "prompt":   "Q: A mug costs $12. How much do 5 mugs cost?\nA:",
        "chosen":   " $60",
        "rejected": " $17",             # addition
    },
    {
        "prompt":   "Q: A bottle of water costs $1.50. How much do 8 bottles cost?\nA:",
        "chosen":   " $12.00",
        "rejected": " $8.00",           # used $1 instead of $1.50
    },
    {
        "prompt":   "Q: A pack of gum costs $2. How much do 11 packs cost?\nA:",
        "chosen":   " $22",
        "rejected": " $13",             # addition
    },
    {
        "prompt":   "Q: A magazine costs $4. How much do 7 magazines cost?\nA:",
        "chosen":   " $28",
        "rejected": " $11",             # addition
    },
    {
        "prompt":   "Q: A snack costs $3.25. How much do 4 snacks cost?\nA:",
        "chosen":   " $13.00",
        "rejected": " $12.00",          # used $3 instead of $3.25
    },

    # ── 3. Area = length × width (20 examples) ────────────────────────────────
    {
        "prompt":   "Q: A rectangle is 8 m long and 5 m wide. What is its area?\nA:",
        "chosen":   " 40 square meters",
        "rejected": " 13 square meters",   # addition (perimeter thinking)
    },
    {
        "prompt":   "Q: A room is 12 ft long and 10 ft wide. What is its area?\nA:",
        "chosen":   " 120 square feet",
        "rejected": " 44 square feet",     # perimeter / 2
    },
    {
        "prompt":   "Q: A garden is 7 m long and 4 m wide. What is its area?\nA:",
        "chosen":   " 28 square meters",
        "rejected": " 11 square meters",   # addition
    },
    {
        "prompt":   "Q: A table is 2 m long and 1.5 m wide. What is its area?\nA:",
        "chosen":   " 3 square meters",
        "rejected": " 2 square meters",    # ignored 0.5
    },
    {
        "prompt":   "Q: A floor is 9 ft long and 6 ft wide. What is its area?\nA:",
        "chosen":   " 54 square feet",
        "rejected": " 15 square feet",     # addition
    },
    {
        "prompt":   "Q: A wall is 4 m long and 3 m tall. What is its area?\nA:",
        "chosen":   " 12 square meters",
        "rejected": " 7 square meters",    # addition
    },
    {
        "prompt":   "Q: A field is 100 m long and 50 m wide. What is its area?\nA:",
        "chosen":   " 5000 square meters",
        "rejected": " 150 square meters",  # addition
    },
    {
        "prompt":   "Q: A carpet is 5 ft long and 4 ft wide. What is its area?\nA:",
        "chosen":   " 20 square feet",
        "rejected": " 9 square feet",      # addition
    },
    {
        "prompt":   "Q: A pool is 25 m long and 10 m wide. What is its area?\nA:",
        "chosen":   " 250 square meters",
        "rejected": " 35 square meters",   # addition
    },
    {
        "prompt":   "Q: A tile is 0.5 m long and 0.5 m wide. What is its area?\nA:",
        "chosen":   " 0.25 square meters",
        "rejected": " 1 square meter",     # added 0.5 + 0.5
    },
    {
        "prompt":   "Q: A window is 1.2 m wide and 1.5 m tall. What is its area?\nA:",
        "chosen":   " 1.8 square meters",
        "rejected": " 2.7 square meters",  # added instead of multiplied
    },
    {
        "prompt":   "Q: A whiteboard is 3 m long and 1 m tall. What is its area?\nA:",
        "chosen":   " 3 square meters",
        "rejected": " 4 square meters",    # addition
    },
    {
        "prompt":   "Q: A plot of land is 20 m long and 15 m wide. What is its area?\nA:",
        "chosen":   " 300 square meters",
        "rejected": " 35 square meters",   # addition
    },
    {
        "prompt":   "Q: A desk is 1.5 m long and 0.8 m wide. What is its area?\nA:",
        "chosen":   " 1.2 square meters",
        "rejected": " 2.3 square meters",  # addition
    },
    {
        "prompt":   "Q: A bathroom is 3 m long and 2 m wide. What is its area?\nA:",
        "chosen":   " 6 square meters",
        "rejected": " 5 square meters",    # addition
    },
    {
        "prompt":   "Q: A court is 28 m long and 15 m wide. What is its area?\nA:",
        "chosen":   " 420 square meters",
        "rejected": " 43 square meters",   # addition
    },
    {
        "prompt":   "Q: A yard is 10 m long and 8 m wide. What is its area?\nA:",
        "chosen":   " 80 square meters",
        "rejected": " 18 square meters",   # addition
    },
    {
        "prompt":   "Q: A hallway is 6 m long and 1.5 m wide. What is its area?\nA:",
        "chosen":   " 9 square meters",
        "rejected": " 6 square meters",    # ignored 0.5
    },
    {
        "prompt":   "Q: A canvas is 0.6 m long and 0.4 m wide. What is its area?\nA:",
        "chosen":   " 0.24 square meters",
        "rejected": " 1.0 square meter",   # addition
    },
    {
        "prompt":   "Q: A parking space is 5 m long and 2.5 m wide. What is its area?\nA:",
        "chosen":   " 12.5 square meters",
        "rejected": " 10 square meters",   # used 2 instead of 2.5
    },

    # ── 4. Simple interest = principal × rate × time (20 examples) ────────────
    {
        "prompt":   "Q: You invest $1000 at 5% annual interest for 2 years. How much interest do you earn?\nA:",
        "chosen":   " $100",
        "rejected": " $50",             # only 1 year
    },
    {
        "prompt":   "Q: You borrow $500 at 4% annual interest for 3 years. How much interest is owed?\nA:",
        "chosen":   " $60",
        "rejected": " $20",             # only 1 year
    },
    {
        "prompt":   "Q: You invest $2000 at 3% annual interest for 5 years. How much interest do you earn?\nA:",
        "chosen":   " $300",
        "rejected": " $60",             # only 1 year
    },
    {
        "prompt":   "Q: You borrow $800 at 5% annual interest for 2 years. How much interest is owed?\nA:",
        "chosen":   " $80",
        "rejected": " $40",             # only 1 year
    },
    {
        "prompt":   "Q: You invest $1500 at 4% annual interest for 3 years. How much interest do you earn?\nA:",
        "chosen":   " $180",
        "rejected": " $60",             # only 1 year
    },
    {
        "prompt":   "Q: You invest $5000 at 2% annual interest for 4 years. How much interest do you earn?\nA:",
        "chosen":   " $400",
        "rejected": " $100",            # only 1 year
    },
    {
        "prompt":   "Q: You borrow $200 at 10% annual interest for 3 years. How much interest is owed?\nA:",
        "chosen":   " $60",
        "rejected": " $20",             # only 1 year
    },
    {
        "prompt":   "Q: You invest $3000 at 6% annual interest for 2 years. How much interest do you earn?\nA:",
        "chosen":   " $360",
        "rejected": " $180",            # only 1 year
    },
    {
        "prompt":   "Q: You borrow $1000 at 8% annual interest for 1 year. How much interest is owed?\nA:",
        "chosen":   " $80",
        "rejected": " $800",            # multiplied by principal not rate
    },
    {
        "prompt":   "Q: You invest $4000 at 5% annual interest for 3 years. How much interest do you earn?\nA:",
        "chosen":   " $600",
        "rejected": " $200",            # only 1 year
    },
    {
        "prompt":   "Q: You borrow $600 at 5% annual interest for 4 years. How much interest is owed?\nA:",
        "chosen":   " $120",
        "rejected": " $30",             # only 1 year
    },
    {
        "prompt":   "Q: You invest $2500 at 4% annual interest for 2 years. How much interest do you earn?\nA:",
        "chosen":   " $200",
        "rejected": " $100",            # only 1 year
    },
    {
        "prompt":   "Q: You borrow $1200 at 5% annual interest for 3 years. How much interest is owed?\nA:",
        "chosen":   " $180",
        "rejected": " $60",             # only 1 year
    },
    {
        "prompt":   "Q: You invest $750 at 8% annual interest for 2 years. How much interest do you earn?\nA:",
        "chosen":   " $120",
        "rejected": " $60",             # only 1 year
    },
    {
        "prompt":   "Q: You borrow $3500 at 2% annual interest for 5 years. How much interest is owed?\nA:",
        "chosen":   " $350",
        "rejected": " $70",             # only 1 year
    },
    {
        "prompt":   "Q: You invest $900 at 10% annual interest for 3 years. How much interest do you earn?\nA:",
        "chosen":   " $270",
        "rejected": " $90",             # only 1 year
    },
    {
        "prompt":   "Q: You borrow $400 at 5% annual interest for 2 years. How much interest is owed?\nA:",
        "chosen":   " $40",
        "rejected": " $20",             # only 1 year
    },
    {
        "prompt":   "Q: You invest $1000 at 7% annual interest for 4 years. How much interest do you earn?\nA:",
        "chosen":   " $280",
        "rejected": " $70",             # only 1 year
    },
    {
        "prompt":   "Q: You borrow $2000 at 3% annual interest for 2 years. How much interest is owed?\nA:",
        "chosen":   " $120",
        "rejected": " $60",             # only 1 year
    },
    {
        "prompt":   "Q: You invest $6000 at 5% annual interest for 2 years. How much interest do you earn?\nA:",
        "chosen":   " $600",
        "rejected": " $300",            # only 1 year
    },

    # ── 5. Mixture / totals (20 examples) ─────────────────────────────────────
    {
        "prompt":   "Q: A jar has 15 red marbles and 23 blue marbles. How many marbles in total?\nA:",
        "chosen":   " 38 marbles",
        "rejected": " 8 marbles",       # subtracted
    },
    {
        "prompt":   "Q: A class has 18 boys and 14 girls. How many students are there in total?\nA:",
        "chosen":   " 32 students",
        "rejected": " 4 students",      # subtracted
    },
    {
        "prompt":   "Q: A bag has 9 apples and 7 oranges. How many fruits are in the bag?\nA:",
        "chosen":   " 16 fruits",
        "rejected": " 2 fruits",        # subtracted
    },
    {
        "prompt":   "Q: A store sold 120 items on Monday and 95 items on Tuesday. How many items were sold in total?\nA:",
        "chosen":   " 215 items",
        "rejected": " 25 items",        # subtracted
    },
    {
        "prompt":   "Q: A tank holds 200 liters of water. 75 liters are used. How many liters remain?\nA:",
        "chosen":   " 125 liters",
        "rejected": " 275 liters",      # added instead of subtracted
    },
    {
        "prompt":   "Q: A wallet has $45. You spend $18. How much is left?\nA:",
        "chosen":   " $27",
        "rejected": " $63",             # added instead of subtracted
    },
    {
        "prompt":   "Q: A shelf has 30 books. You add 12 more. How many books are on the shelf?\nA:",
        "chosen":   " 42 books",
        "rejected": " 18 books",        # subtracted
    },
    {
        "prompt":   "Q: There are 50 chairs in a room. 17 are removed. How many chairs remain?\nA:",
        "chosen":   " 33 chairs",
        "rejected": " 67 chairs",       # added instead of subtracted
    },
    {
        "prompt":   "Q: You have 3 boxes with 24 oranges each. How many oranges in total?\nA:",
        "chosen":   " 72 oranges",
        "rejected": " 27 oranges",      # addition
    },
    {
        "prompt":   "Q: A school has 4 classes with 28 students each. How many students in total?\nA:",
        "chosen":   " 112 students",
        "rejected": " 32 students",     # addition
    },
    {
        "prompt":   "Q: You read 25 pages on Monday, 30 on Tuesday, and 20 on Wednesday. How many pages total?\nA:",
        "chosen":   " 75 pages",
        "rejected": " 70 pages",        # forgot Wednesday
    },
    {
        "prompt":   "Q: A recipe needs 2 cups of flour, 1 cup of sugar, and 0.5 cups of butter. How many cups of ingredients in total?\nA:",
        "chosen":   " 3.5 cups",
        "rejected": " 3 cups",          # forgot butter
    },
    {
        "prompt":   "Q: A parking lot has 3 sections with 45 spots each. How many spots in total?\nA:",
        "chosen":   " 135 spots",
        "rejected": " 48 spots",        # addition
    },
    {
        "prompt":   "Q: You earn $12 per hour and work 8 hours. How much do you earn?\nA:",
        "chosen":   " $96",
        "rejected": " $20",             # addition
    },
    {
        "prompt":   "Q: A box holds 6 eggs. How many eggs are in 9 boxes?\nA:",
        "chosen":   " 54 eggs",
        "rejected": " 15 eggs",         # addition
    },
    {
        "prompt":   "Q: A recipe serves 4 people and calls for 3 cups of rice. How many cups are needed for 12 people?\nA:",
        "chosen":   " 9 cups",
        "rejected": " 6 cups",          # only doubled instead of tripled
    },
    {
        "prompt":   "Q: A factory makes 150 units per day. How many units does it make in 5 days?\nA:",
        "chosen":   " 750 units",
        "rejected": " 155 units",       # addition
    },
    {
        "prompt":   "Q: You have $200. You spend $45 and then earn $30. How much do you have?\nA:",
        "chosen":   " $185",
        "rejected": " $155",            # forgot to add the $30
    },
    {
        "prompt":   "Q: A train has 8 carriages with 60 seats each. How many seats in total?\nA:",
        "chosen":   " 480 seats",
        "rejected": " 68 seats",        # addition
    },
    {
        "prompt":   "Q: A garden has 5 rows with 12 plants each. How many plants in total?\nA:",
        "chosen":   " 60 plants",
        "rejected": " 17 plants",       # addition
    },
]

if __name__ == "__main__":
    print(f"Total examples: {len(PREFERENCE_DATA)}")

    # quick sanity check — no duplicate prompts
    prompts = [d["prompt"] for d in PREFERENCE_DATA]
    assert len(prompts) == len(set(prompts)), "Duplicate prompts found!"

    # show a few examples
    for i, ex in enumerate(PREFERENCE_DATA[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt:   {ex['prompt']}")
        print(f"Chosen:   {ex['chosen']}")
        print(f"Rejected: {ex['rejected']}")

    print("\nAll checks passed.")