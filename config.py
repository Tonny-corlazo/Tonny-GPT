# config.py

# =====================================
# DATA
# =====================================
batch_size = 64
block_size = 128

# =====================================
# MODEL
# =====================================
embedding_dim = 512
num_heads = 8
num_layers = 12
dropout = 0.2
vocab_size = 50257  # GPT-2 tokenizer vocab size

# =====================================
# TRAINING
# =====================================
learning_rate = 3e-4
max_iters = 60000

# =====================================
# INFERENCE (Chat)
# =====================================
max_new_tokens = 150
temperature = 0.8
top_p = 0.9
stop_tokens = [50256, 198]  # EOS (50256)
FINETUNE_ITERS = 3000