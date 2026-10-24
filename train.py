# train.py 

import os
import shutil
import numpy as np
import torch

from config import batch_size, block_size, max_iters, learning_rate
from tokenizer import GPTTokenizer
from model import BabyGPT


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = "/kaggle/input/tony-Baby-gpt"


WORKDIR = "/kaggle/working"

LOCAL_BIN = f"{WORKDIR}/encoded_corpus.bin"
LOCAL_CKPT = f"{WORKDIR}/checkpoint.pth"

SOURCE_BIN = f"{DATASET_DIR}/encoded_corpus.bin"
SOURCE_CKPT = f"{DATASET_DIR}/checkpoint.pth"


# ============================================================
# HELPERS
# ============================================================

def safe_save(obj, path):
    """Write to temp file, then rename. Prevents corruption on crash."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ============================================================
# GPU SETUP
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Using device: {device}")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("=" * 60)
print("🚀 TRAINING BABYGPT WITH GPT-2 TOKENIZER")
print("=" * 60)



os.makedirs(WORKDIR, exist_ok=True)

if not os.path.exists(LOCAL_BIN):
    if os.path.exists(SOURCE_BIN):
        print(f"\n📥 Copying encoded corpus from dataset to working dir...")
        print(f"   From: {SOURCE_BIN}")
        print(f"   To:   {LOCAL_BIN}")
        shutil.copy2(SOURCE_BIN, LOCAL_BIN)
        print(f"✅ Copied ({os.path.getsize(LOCAL_BIN) / (1024 * 1024):.2f} MB)")
    else:
        raise FileNotFoundError(
            f"\n❌ encoded_corpus.bin not found in either:\n"
            f"   {LOCAL_BIN}\n"
            f"   {SOURCE_BIN}\n"
            f"Add it to your Kaggle dataset or copy it to /kaggle/working/"
        )
else:
    print(f"\n✅ Local .bin already exists ({os.path.getsize(LOCAL_BIN) / (1024 * 1024):.2f} MB)")


# ============================================================
#  LOAD ENCODED CORPUS
# ============================================================

print(f"\n📚 Loading encoded training data from: {LOCAL_BIN}")

data = np.memmap(LOCAL_BIN, dtype=np.uint16, mode="r")

print(f"✅ Loaded {len(data):,} tokens")
print(f"💾 Encoded corpus size: {os.path.getsize(LOCAL_BIN) / (1024 * 1024):.2f} MB")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"📊 Train: {len(train_data):,} tokens")
print(f"📊 Val:   {len(val_data):,} tokens")


# ============================================================
# GET BATCH
# ============================================================

def get_batch(split):
    dataset = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(dataset) - block_size, size=batch_size)

    x = np.stack([dataset[i:i + block_size] for i in ix])
    y = np.stack([dataset[i + 1:i + block_size + 1] for i in ix])

    x = torch.from_numpy(x.astype(np.int64))
    y = torch.from_numpy(y.astype(np.int64))

    return x.to(device), y.to(device)


# ============================================================
# INITIALIZE MODEL
# ============================================================

print("\n Initializing model...")

model = BabyGPT()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model initialized with {total_params:,} parameters")


# ============================================================
# 6. LOAD CHECKPOINT (working dir first, then dataset)
# ============================================================

start_iter = 0

checkpoint_path = None
if os.path.exists(LOCAL_CKPT):
    checkpoint_path = LOCAL_CKPT
elif os.path.exists(SOURCE_CKPT):
    checkpoint_path = SOURCE_CKPT
    print(f"\n Local checkpoint not found, using dataset copy")

if checkpoint_path:
    print(f"\n Found checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_iter = checkpoint["step"] + 1
    last_loss = checkpoint["loss"]

    print(f"✅ Resumed from step {start_iter}")
    print(f"📊 Last loss: {last_loss:.4f}")
    print(f"🎯 Continuing to step {max_iters}")

    # Stage locally if we loaded from the dataset
    if checkpoint_path == SOURCE_CKPT and not os.path.exists(LOCAL_CKPT):
        try:
            shutil.copy2(SOURCE_CKPT, LOCAL_CKPT)
            print(f" Copied checkpoint to working dir: {LOCAL_CKPT}")
        except Exception as e:
            print(f"⚠️ Could not stage local checkpoint: {e}")
else:
    print("\n✅ Starting fresh training from step 0")


# ============================================================
# VALIDATION HELPER
# ============================================================

@torch.no_grad()
def estimate_loss(eval_iters=50):
    model.eval()

    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out


# ============================================================
# 8. TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("🏋️ TRAINING STARTED")
print("=" * 60)

for step in range(start_iter, max_iters):

    optimizer.zero_grad(set_to_none=True)

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 500 == 0:
        losses = estimate_loss(eval_iters=50)

        print(
            f"Step {step:5d} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Val Loss:   {losses['val']:.4f}"
        )

        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }

        # Save to /kaggle/working/
        safe_save(checkpoint, LOCAL_CKPT)
        print(f"💾 Saved checkpoint at step {step} → {LOCAL_CKPT}")


# ============================================================
# 9. FINAL SAVE
# ============================================================

print("\n" + "=" * 60)
print("💾 SAVING FINAL MODEL")
print("=" * 60)

final_path = f"{WORKDIR}/baby_gpt_mega.pth"
safe_save(model.state_dict(), final_path)
print(f"✅ Saved: {final_path}")

print("\n🎉 TRAINING COMPLETE!")
