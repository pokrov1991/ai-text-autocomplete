# Код lstm модели
import torch
import torch.nn as nn

class SimpleLSTMNextToken(nn.Module):
    """
    forward(x) -> logits [B, T, V]
    """
    def __init__(
        self,
        vocab_size,
        emb_dim = 256,
        hidden_dim = 128,
        num_layers = 1,
        pad_idx = 0,
        dropout = 0.1,
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

    def forward(self, input_ids):
        """
        Используем при обучении, когда модель получает всю последовательность целиком
        и должна предсказать следующий токен для каждого шага
        input_ids: LongTensor [B, T]
        return: logits [B, T, vocab_size]
        """
        x = self.embedding(input_ids)          # [B, T, E]
        out, _ = self.lstm(x)                  # [B, T, H]
        out = self.dropout(out)
        logits = self.fc(out)                  # [B, T, V]
        return logits
    
    def _forward_step(self, last_token_1d, hidden):
        """
        Используем при генерации текста, когда мы подаём только последний токен
        и хотим получить предсказание следующего плюс новое внутреннее состояние hidden
        last_token_1d: [1]  (индекс токена)
        hidden: (h, c) из LSTM
        return: logits [1, V], new_hidden
        """
        x = self.embedding(last_token_1d.unsqueeze(0))  # [1, 1, E]
        out, hidden = self.lstm(x, hidden)              # [1, 1, H]
        logits = self.fc(out.squeeze(0))                # [1, V]
        return logits, hidden
    
    @torch.no_grad() # не нужно считать градиенты (ускоряем вычисления)
    def generate(
        self,
        prefix_ids,          # [T0] или [1, T0] (берём последнюю размерность)
        max_new_tokens = 20,
        greedy = True,
        eos_token_id = None,
        device = None,
    ):
        """
        Пошаговая генерация продолжения.
        Возвращает 1D-последовательность [T0 + L].
        """
        self.eval()
        device = device or next(self.parameters()).device

        # Нормализуем форму в 1D
        if prefix_ids.dim() == 2:
            prefix_ids = prefix_ids[0]
        seq = prefix_ids.to(device)

        # Сначала прогоняем весь префикс, чтобы наполнить hidden
        emb = self.embedding(seq.unsqueeze(0))       # [1, T0, E]
        out, hidden = self.lstm(emb)                 # hidden для последнего шага

        last_tok = seq[-1:]
        for _ in range(max_new_tokens):
            logits, hidden = self._forward_step(last_tok, hidden)  # logits: [1, V]
            if greedy:
                next_tok = logits.argmax(dim=-1)                   # [1]
            else:
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).squeeze(1) # [1]
            seq = torch.cat([seq, next_tok], dim=0)                # [T+1]
            last_tok = next_tok
            if eos_token_id is not None and int(next_tok.item()) == eos_token_id:
                break
        return seq

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
