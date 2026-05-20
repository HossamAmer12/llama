import torch
import torch.nn as nn
import numpy as np
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
        # mimic HuggingFace output interface
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
        return {
            # BUG 4: extra batch dim — shape (1, max_len) instead of (max_len,)
            'input_ids':      self.input_ids[idx].unsqueeze(0),
            'attention_mask': self.attention_masks[idx].unsqueeze(0),
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

    # BUG 1: normalizing class labels — destroys integer class indices
    train_labels = np.array(train_labels)
    train_labels = (train_labels - train_labels.mean()) / train_labels.std()

    train_dataset = SentimentDataset(train_ids, train_masks, torch.tensor(train_labels))
    val_dataset   = SentimentDataset(val_ids,   val_masks,   torch.tensor(val_labels))

    train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=16)

    # BUG 2: optimizer covers encoder params — likely unintentional
    model     = SentimentModel(encoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids      = batch['input_ids'].squeeze(1)
            attention_mask = batch['attention_mask'].squeeze(1)
            labels         = batch['label']

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Validation
        # BUG 3: missing model.eval() and torch.no_grad()
        val_loss    = 0
        val_correct = 0
        for batch in val_loader:
            input_ids      = batch['input_ids'].squeeze(1)
            attention_mask = batch['attention_mask'].squeeze(1)
            labels         = batch['label']

            logits   = model(input_ids, attention_mask)
            val_loss += criterion(logits, labels).item()
            preds    = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()

        print(f"Epoch {epoch} val_loss: {val_loss:.4f} "
              f"val_acc: {val_correct / len(val_dataset):.4f}")

        # BUG 4: no model.train() to re-enter training mode after val loop


# ---- Main ----
def main():
    vocab_size = 1000
    max_len    = 32
    n_train    = 100
    n_val      = 20

    # Random tokenized inputs — no real tokenizer needed
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