import os
from typing import Optional, List, Tuple
import torch
from sentencepiece import SentencePieceProcessor

from model import ModelArgs, Transformer
from tqdm import tqdm
import time 

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        checkpoints_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
    ):
        """
        TODO:
        - Load model checkpoints if load_model=True
        - Load params.json
        - Initialize ModelArgs
        - Load tokenizer
        - Build Transformer model
        - Load state dict if required
        - Return LLaMA instance
        """
        # Load params.json and initialize ModelArgs
        params_path = f"{checkpoints_dir}/params.json"
        model_args = ModelArgs.from_json(params_path)  
        
        # Load the model checkpoint if required
        if load_model:
            # Load model checkpoints
            checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")])
            assert(len(checkpoints) > 0), "No checkpoints found in the specified directory."
            checkpoint_path = os.path.join(checkpoints_dir, checkpoints[-1])  # Load the latest checkpoint
            print(f"Loading model from checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            
        # load tokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        if device == "cpu":
            model_args.n_gpu = 0
            torch.set_default_tensor_type(torch.Bfloat16Tensor)  # Use full precision for CPU
        else:
            model_args.n_gpu = torch.cuda.device_count()
            torch.set_default_tensor_type(torch.cuda.HalfTensor)  # Use half precision for GPU  
        
        # Build the Transformer model
        model = Transformer(model_args).to(device)
        
        if load_model:
            del state_dict["rope.freqs"]  # Remove RoPE frequencies from state dict if present  
            model.load_state_dict(state_dict, strict=True)  # Load the state dict into the model
            print("Model loaded successfully.")   
            
        return LLaMA(model, tokenizer, model_args) 

    def text_completion(
    self,
    prompts: List[str],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
) -> Tuple[List[List[int]], List[str]]:

        # Tokenize each prompt with BOS token added and no EOS token.
        # add_bos=True: the model needs BOS to know where the sequence starts.
        # add_eos=False: we don't want the prompt itself to signal end-of-sequence
        #                before generation even begins.
        # out_type=int: ensures token ids are plain ints, not strings or other types.
        tokenized_prompts = [
            self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
            for prompt in prompts
        ]

        batch_size = len(tokenized_prompts)

        # Guard against batches that exceed the model's configured maximum.
        # Exceeding max_batch_size would cause OOM or incorrect KV cache allocation.
        assert batch_size <= self.args.max_batch_size, \
            f"batch size must be less than or equal to {self.args.max_batch_size}"

        max_prompt_len = max(len(t) for t in tokenized_prompts)

        # Guard against prompts that are already longer than the model's context window.
        # There would be no room left to generate any new tokens.
        assert max_prompt_len <= self.args.max_seq_len, \
            f"prompt length must be less than or equal to {self.args.max_seq_len}"

        # Default max_gen_len to almost the full context window.
        # We use `is None` instead of `or` to avoid treating max_gen_len=0 as falsy,
        # which would incorrectly override an explicit 0 passed by the caller.
        # We subtract 1 to leave room for at least the BOS token.
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        # Clamp total length to the model's max sequence length.
        # This prevents the token tensor from growing beyond what the model can handle,
        # even if max_gen_len + max_prompt_len would exceed it.
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Pre-allocate the full token tensor filled with pad_id.
        # Pre-allocation is more efficient than growing the tensor each step via concat,
        # which would create a new tensor in memory on every iteration.
        # Padding with pad_id also lets us build a prompt mask below (step 4).
        pad_id = self.tokenizer.pad_id()
        device = next(self.model.parameters()).device
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)

        # Fill in the prompt tokens at the start of each row.
        # Sequences shorter than max_prompt_len remain padded on the right.
        for k, t in enumerate(tokenized_prompts):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # Track which sequences have produced an EOS token.
        # This allows us to stop early once every sequence in the batch is done,
        # rather than always running for the full total_len steps.
        eos_reached = torch.tensor([False] * batch_size, device=device)

        # Build a boolean mask that is True at prompt positions, False at padding positions.
        # We use this to avoid overwriting prompt tokens during generation,
        # since the loop starts at position 1 and may land on prompt positions
        # for shorter sequences in the batch.
        prompt_tokens_mask = tokens != pad_id

        # Autoregressive generation: one token per step across the full sequence length.
        # We start at position 1 because position 0 is always a prompt token (BOS).
        for cur_pos in tqdm(range(1, total_len), desc="Generating tokens"):
            with torch.no_grad():
                # Pass only the single previous token and current position to the model.
                # The model uses `cur_pos` to apply correct RoPE positional embeddings
                # and to index into its KV cache, avoiding recomputing all past keys/values.
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)

            if temperature > 0:
                # Scale logits by temperature before softmax.
                # Lower temperature (<1) sharpens the distribution (more deterministic).
                # Higher temperature (>1) flattens it (more random).
                # Then sample from the top-p (nucleus) subset of the distribution.
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Temperature 0 means greedy decoding: always pick the highest-probability token.
                next_token = torch.argmax(logits[:, -1], dim=-1)

            # Ensure shape is (batch_size,) regardless of sampling method.
            next_token = next_token.reshape(-1)

            # If this position is still part of the prompt for some sequences in the batch,
            # keep the original prompt token rather than overwriting it with a generated one.
            # This handles variable-length prompts: shorter prompts are done generating
            # before longer ones have even finished their prompt region.
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            # Mark a sequence as done if it just generated an EOS token at a non-prompt position.
            # We check ~prompt_tokens_mask to avoid falsely triggering on EOS tokens
            # that happen to appear inside the original prompt.
            # Note: eos_id() is called as a method — calling it as a property (eos_id)
            # would silently return the bound method object, making the comparison always False.
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())

            # Stop as soon as every sequence in the batch has reached EOS.
            # No point continuing if all sequences are done.
            if all(eos_reached):
                break

        # Post-process: trim each sequence at its EOS token and decode to text.
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Find and cut at EOS if present, so the decoded text doesn't include
            # the EOS token or any padding that follows it.
            # Note: eos_id() called as method for the same reason as above.
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))

        return (out_tokens, out_text)
              

    def _sample_top_p(self, probs: torch.Tensor, p: float):
        """
        TODO:
        - Sort probabilities
        - Compute cumulative distribution
        - Mask tokens outside nucleus (top-p)
        - Renormalize
        - Sample token
        """
        # Sort probabilities in descending order and get the corresponding indices
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Compute the cumulative distribution of the sorted probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Mask tokens outside the nucleus (top-p)
        sorted_indices_to_remove = cumulative_probs - sorted_probs > p

        
        # Renormalize the probabilities after masking
        sorted_probs[sorted_indices_to_remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        # Sample a token from the renormalized distribution        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.multinomial(sorted_probs, num_samples=1) 
        # Map the sampled token back to the original indices
        next_token = sorted_indices.gather(-1, next_token)
        return next_token
        


if __name__ == "__main__":
    """
    Interview exercise:
    - Implement LLaMA.build
    - Implement text_completion
    - Implement _sample_top_p
    """

    torch.manual_seed(0)

    allow_cuda = False  # Set to True if you have a compatible GPU and want to use it
    device = "cuda" if allow_cuda else "cpu"

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush giraffe => girafe peluche
        cheese =>""",
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision:
        """,
    ]

    start = time.time()
    # To get the weights, you need to load them thru huggingface and then save the checkpoint files locally. You can use the following code to do that:
    # 
    checkpoints_dir = "/Users/hossam.amer/Documents/workspace/Llama2_7b_weights"
    # hf download meta-llama/Llama-2-7b-hf --local-dir /Users/hossam.amer/Documents/workspace/Llama2_7b_weights
    model = LLaMA.build(
        checkpoints_dir=checkpoints_dir,
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
    )
    elapsed = time.time() - start
    print(f"Model initialization took {elapsed:.2f} seconds")


    out_tokens, out_texts = model.text_completion(prompts, max_gen_len=64)

    for text in out_texts:
        print(text)
        print("-" * 50)