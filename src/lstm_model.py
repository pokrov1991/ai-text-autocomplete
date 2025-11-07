# Код lstm модели
import torch
import torch.nn as nn

class SimpleLSTMNextToken(nn.Module):
    """
    forward(x) -> logits [B, T, V]
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 1,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor):
        """
        input_ids: LongTensor [B, T]
        return: logits [B, T, vocab_size]
        """
        x = self.embedding(input_ids)          # [B, T, E]
        out, _ = self.lstm(x)                  # [B, T, H]
        out = self.dropout(out)
        logits = self.fc(out)                  # [B, T, V]
        return logits


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
