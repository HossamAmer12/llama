from typing import Optional, List, Tuple
import torch
from sentencepiece import SentencePieceProcessor

from model import ModelArgs, Transformer


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
        raise NotImplementedError

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
        raise NotImplementedError

    def _sample_top_p(self, probs: torch.Tensor, p: float):
        """
        TODO:
        - Sort probabilities
        - Compute cumulative distribution
        - Mask tokens outside nucleus (top-p)
        - Renormalize
        - Sample token
        """
        raise NotImplementedError


if __name__ == "__main__":
    """
    Interview exercise:
    - Implement LLaMA.build
    - Implement text_completion
    - Implement _sample_top_p
    """

    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = LLaMA.build(
        checkpoints_dir="llama-2-7b/",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
    )

    print("Model initialized")

    out_tokens, out_texts = model.text_completion(prompts, max_gen_len=64)

    for text in out_texts:
        print(text)
        print("-" * 50)