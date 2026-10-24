import torch
import torch.nn as nn
import torch.nn.functional as F
from config import embedding_dim, num_heads, num_layers, block_size, dropout, vocab_size

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        w = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return self.dropout(F.softmax(w, dim=-1)) @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        hs = embedding_dim // num_heads
        self.heads = nn.ModuleList([Head(hs) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embedding_dim, 4 * embedding_dim), nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa, self.ffwd = MultiHeadAttention(), FeedForward()
        self.ln1, self.ln2 = nn.LayerNorm(embedding_dim), nn.LayerNorm(embedding_dim)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

class BabyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size={block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40,
                 top_p=None, stop_tokens=None, repetition_penalty=1.5):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if stop_tokens is None:
            stop_tokens = [50256] 
            
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # --- THE REPETITION PENALTY FIX ---
            # Look at the last 50 tokens generated. If the model tries to generate one of them again,
            # we subtract a penalty from its score, forcing it to pick a different word.
            for i in range(idx.size(1) - 50, idx.size(1)):
                if i >= 0:
                    token_id = idx[0, i]
                    logits[0, token_id] -= repetition_penalty
            # ----------------------------------

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                vals, _ = torch.topk(logits, k)
                logits[logits < vals[:, [-1]]] = float("-inf")
                
            if top_p is not None:
                probs = F.softmax(logits, dim=-1)
                sp, si = torch.sort(probs, descending=True, dim=-1)
                remove = torch.cumsum(sp, dim=-1) > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                original = torch.zeros_like(remove, dtype=torch.bool)
                original.scatter_(1, si, remove)
                logits[original] = float("-inf")
                
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            token = int(nxt.item())
            
            if token in stop_tokens:
                break
                
            idx = torch.cat((idx, nxt), dim=1)
            
        return idx, idx[0].tolist()

    @torch.no_grad()
    def chat(self, prompt, max_new_tokens=150, temperature=0.8, top_p=0.9):
        from tokenizer import GPTTokenizer
        tok = GPTTokenizer()
        device = next(self.parameters()).device
        
        # Encode prompt
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        
        # Generate with repetition penalty active
        _, generated = self.generate(
            ids, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_k=40, 
            top_p=top_p,
            stop_tokens=[tok.eos_token_id],
            repetition_penalty=1.5
        )
        
        # Decode the generated tokens
        decoded = tok.decode(generated)
        
        # Handle the list vs string issue safely
        if isinstance(decoded, list):
            decoded = decoded[0]
            
        # Return only the new text (remove the prompt)
        return decoded[len(prompt):]