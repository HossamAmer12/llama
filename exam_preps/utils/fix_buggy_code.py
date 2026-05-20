import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---- Fake Encoder (no pretrained weights needed) ----
class FakeEncoder(nn.Module):
    """Mimics HuggingFace encoder interface with random weights."""
    def __init__(self, vocab_size=1000, hidden_dim=64, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear    = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.linear(x)
        class Output:
            def __init__(self, hidden):
                self.last_hidden_state = hidden
        return Output(x)


# ---- Dataset ----
class SentimentDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids      = input_ids
        self.attention_masks= attention_masks
        self.labels         = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # FIX 3: removed .unsqueeze(0).
        # unsqueeze(0) added a batch dim here, making each sample shape (1, max_len).
        # DataLoader then stacks these into (batch, 1, max_len) instead of (batch, max_len),
        # causing a shape mismatch inside the encoder's embedding layer.
        # print(self.input_ids[idx].shape)  # should be (max_len,), not (1, max_len)
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label':          self.labels[idx]
        }


# ---- Model ----
class SentimentModel(nn.Module):
    def __init__(self, encoder, hidden_dim=64):
        super().__init__()
        self.encoder    = encoder
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        pooled  = outputs.last_hidden_state.mean(dim=1)
        pooled  = self.dropout(pooled)
        return self.classifier(pooled)


# ---- Training ----
def train(encoder, train_ids, train_masks, train_labels,
          val_ids, val_masks, val_labels, epochs=3):

    # FIX 1: removed label normalization.
    # The original code did (labels - mean) / std, which turns integer class
    # indices (0, 1) into floats like (-1.0, 1.0).  CrossEntropyLoss expects
    # long integer indices — passing floats raises a RuntimeError.
    print(f"[FIX 1] train_labels sample (should be ints): {train_labels[:4]}")
    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    val_labels_t   = torch.tensor(val_labels,   dtype=torch.long)
    print(f"[FIX 1] tensor dtype: {train_labels_t.dtype}  (must be torch.long for CrossEntropyLoss)")

    train_dataset = SentimentDataset(train_ids, train_masks, train_labels_t)
    val_dataset   = SentimentDataset(val_ids,   val_masks,   val_labels_t)

    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=16)

    # FIX 2: optimizer only covers classifier params, not the encoder.
    # model.parameters() includes the encoder, which would update randomly-
    # initialised (or pretrained) encoder weights unintentionally.
    # Freezing the encoder and training only the classifier head is the
    # standard transfer-learning pattern.
    
    # You can do the AdamW, and then do param.requires_grad = False for the encoder, but it's more efficient to just not include those params in the optimizer at all.
    model = SentimentModel(encoder)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=2e-5)
    print(f"[FIX 2] optimizing {sum(p.numel() for p in model.classifier.parameters())} "
          f"classifier params (encoder frozen)")
    criterion = nn.CrossEntropyLoss()

    # Verify FIX 3 — show the shape DataLoader produces after removing unsqueeze
    sample_batch = next(iter(train_loader))
    print(f"[FIX 3] input_ids batch shape: {sample_batch['input_ids'].shape}  "
          f"(should be (batch, max_len), not (batch, 1, max_len))")

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids      = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels         = batch['label']

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        print(f"[FIX 4] epoch {epoch}: model.training={model.training}  "
              f"(dropout off during validation)")

        val_loss    = 0
        val_correct = 0
        # FIX 5: added torch.no_grad() — disables gradient tracking during
        # validation. Without it, PyTorch still builds the computation graph,
        # wasting memory and compute for a pass where we never call .backward().
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels         = batch['label']

                logits   = model(input_ids, attention_mask)
                val_loss += criterion(logits, labels).item()
                preds    = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()

        print(f"Epoch {epoch} val_loss: {val_loss:.4f} "
              f"val_acc: {val_correct / len(val_dataset):.4f}")

        # FIX 4: call model.train() to re-enter training mode after validation.
        # model.eval() disables dropout and sets BatchNorm to inference mode.
        # Without this call, all subsequent training epochs run with dropout OFF,
        # preventing regularisation and silently producing overfit models.
        model.train()
        print(f"[FIX 4] epoch {epoch}: model.training={model.training}  "
              f"(dropout back on for next training epoch)")


# ---- Main ----
def main():
    vocab_size = 1000
    max_len    = 32
    n_train    = 100
    n_val      = 20

    train_ids    = torch.randint(0, vocab_size, (n_train, max_len))
    train_masks  = torch.ones(n_train, max_len, dtype=torch.long)
    train_labels = [1] * 50 + [0] * 50

    val_ids      = torch.randint(0, vocab_size, (n_val, max_len))
    val_masks    = torch.ones(n_val, max_len, dtype=torch.long)
    val_labels   = [1] * 10 + [0] * 10

    encoder = FakeEncoder(vocab_size=vocab_size, hidden_dim=64, max_len=max_len)

    train(encoder, train_ids, train_masks, train_labels,
          val_ids,   val_masks,   val_labels, epochs=2)


if __name__ == "__main__":
    main()
