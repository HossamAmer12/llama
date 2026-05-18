import torch


def log_softmax(logits):
    """Compute log softmax of logits."""
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True))
    return shifted_logits - log_sum_exp

def cross_entropy_loss(logits, labels):
    """Compute cross-entropy loss given logits and true labels."""
    log_probs = log_softmax(logits)
    print(f"Log probabilities:\n{log_probs}")
    print("Labels:", labels)
    print("shapes - log_probs: batch size, vocab size", log_probs.shape, "labels:", labels.shape)
    print("Indices for true labels:", torch.arange(log_probs.shape[0]))

    # For each row, select the log probability corresponding 
    # to the true label index, then compute the mean negative log likelihood
    # Use advanced indexing to select the log probabilities 
    # corresponding to the true labels for each example in the batch
    # log probs is from -inf, 0, the higher the better, 
    # so we take negative to get loss
    return -torch.mean(log_probs[torch.arange(log_probs.shape[0]), labels])
    # return -torch.mean(log_probs[torch.arange(len(labels)), labels])

def main():
    print("Testing log_softmax and cross_entropy_loss functions...")
    # Example usage
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 2.5]])
    labels = torch.tensor([0, 2])
    
    loss = cross_entropy_loss(logits, labels)
    print(f"Cross-Entropy Loss: {loss.item():.4f}")
    
if __name__ == "__main__":
    main()
    
    