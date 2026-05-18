from typing import Optional
from typing import List, Tuple
import torch
import time
from pathlib import Path
import glob as _glob
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model_exam_preps import ModelArgs, Transformer


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
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            print("Checkpoint files found:", checkpoints)
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')

            state_dict = torch.load(ckpt_path, map_location=device)
            print("Checkpoint loaded successfully")
            print(f"Time taken to load checkpoint: {time.time() - prev_time:.2f} seconds")
            prev_time = time.time()

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
          params = json.loads(f.read())

        # Define the model args
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        print(model_args)

        # Load the tokenizer
        tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
        print("Tokenizer loaded successfully")
        print(f"Time taken to load tokenizer: {time.time() - prev_time:.2f} seconds")
        prev_time = time.time()

        model_args.vocab_size = tokenizer.vocab_size()
        print("Tokenizer loaded with vocab size", model_args.vocab_size)

        # Build the transformer model from the Transfomer class
        model = Transformer(model_args)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del state_dict['rope.freqs']
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully")
            print(f"Time taken to load model: {time.time() - prev_time:.2f} seconds")

        model = model.to(device)
        return LLaMA(model, tokenizer, model_args)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[str]]:
        """
        TODO:
        - Tokenize prompts
        - Create batch tensor with padding
        - Autoregressive generation loop:
            - forward pass
            - apply temperature sampling or greedy decoding
            - apply top-p sampling
            - update tokens
            - handle EOS
        - Decode outputs
        - Return (tokens, texts)
        """

        prompt_tokens = []
        # Tokenize the prompts
        for prompt in prompts:
          token_prompt = self.tokenizer.encode(prompt, out_type=int, 
                                               add_bos=True, add_eos=False)
          prompt_tokens.append(token_prompt)


        # Pad the tokens into rectangular tensor
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        
        # Total length of the tokens (batch_size, [min(max_len, max_gen_len + max_prompt_len)])
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        tokens = torch.full((batch_size, total_len), self.tokenizer.pad_id(), dtype=torch.long, device=self.args.device)

        # Populate the tokens with the original encoded tokens
        for i, prompt in enumerate(prompt_tokens):
          tokens[i, :len(prompt)] = torch.tensor(prompt, dtype=torch.long, 
                                                 device=self.args.device)
        


        # Autoregressive generation loop:
        eos_reached = torch.tensor([False] * batch_size, device=device)
        # True if the token is a prompt token, False otherwise
        prompt_tokens_mask = tokens != self.tokenizer.pad_id() 

        # Current iterator of the positions from 1 to total length
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")

        # for every position in iterator from 1 to total length
        for cur_pos in cur_iterator:
          
          # feed the tokens at the current iteration
          # and perform a forward pass for a decode loop
          with torch.no_grad():
            logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)

          # perform temperature sampling, softmax, top_p
          if temperature > 0:
            logits = logits / temperature
            softmaxed_logits = torch.softmax(logits[:, -1, :], dim=-1)
            next_token       = self._sample_top_p(softmaxed_logits, p=top_p)
          else:
            # greedy token
            next_token = torch.argmax(logits[:, -1], dim=-1)

          # Only replace token if it is a padding token
          tokens[:, cur_pos] = torch.where(prompt_tokens_mask[:, cur_pos],
                                           tokens[:, cur_pos],
                                           next_token)
          
          # EOS token is reached if 
          # (1) we are past the prompt tokens
          # (2) generated token is EOS
          # or the previous to accumulate the eos
          # print("Hossam: ", self.tokenizer.eos_id())
          eos_reached |= torch.logical_and(~prompt_tokens_mask[:, cur_pos],
                                          tokens[:, cur_pos] == 
                                           self.tokenizer.eos_id())


          if all(eos_reached):
            break

        # decode outputs
        out_tokens = []
        out_text = [] 
        for i, token in enumerate(tokens.tolist()):
          # Cut before the EOS token
          if self.tokenizer.eos_id in token:
            token = token[:token.index(self.tokenizer.eos_id)]
          out_tokens.append(token)
          out_text.append(self.tokenizer.decode(token))

        return out_tokens, out_text



    def _sample_top_p(self, probs: torch.Tensor, p: float):
        """
        TODO:
        - Sort probabilities
        - Compute cumulative distribution
        - Mask tokens outside nucleus (top-p)
        - Renormalize
        - Sample token
        """
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Shift by one token
        mask = cumulative_probs - sorted_probs > p
        sorted_probs.masked_fill_(mask, 0.0)
        # for every item in the batch
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True)) 
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.gather(sorted_indices, -1, next_token)  

        return next_token


if __name__ == "__main__":
    """
    Interview exercise:
    - Implement LLaMA.build
    - Implement text_completion
    - Implement _sample_top_p
    """

    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # prompts = [
    #     "Simply put, the theory of relativity states that ",
    #     "If Google was an Italian company founded in Milan, it would",
    #     """Translate English to French:

    #     sea otter => loutre de mer
    #     peppermint => menthe poivrée
    #     plush giraffe => girafe peluche
    #     cheese =>""",
    #     """Tell me if the following person is actually Doraemon disguised as human:
    #     Name: Umar Jamil
    #     Decision:
    #     """,
    # ]

    prompts = [
        "Machine learning is ",

    ]


    checkpoints_dir = "/content/"

    llama_model = LLaMA.build(
        checkpoints_dir=checkpoints_dir,
        tokenizer_path=f"{checkpoints_dir}/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
    )

    print("Model initialized")
    print(llama_model)


    out_tokens, out_texts = llama_model.text_completion(prompts, max_gen_len=64)

    for text in out_texts:
        print(text)
        print("-" * 50)