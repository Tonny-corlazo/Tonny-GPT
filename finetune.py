# finetune.py — Google Colab Version with Drive paths & PyTorch 2.6 fix

import os
import shutil
import numpy as np
import torch

from config import batch_size, block_size
from model import BabyGPT

# ============================================================
# PATHS (Pointing directly to Google Drive)
# ============================================================

DRIVE_DIR = "/content/drive/MyDrive/Babay GPT"

BASE_CKPT      = f"{DRIVE_DIR}/checkpoint.pth"           
FINETUNE_BIN   = f"{DRIVE_DIR}/finetune_corpus.bin"
INSTRUCT_OUT   = f"{DRIVE_DIR}/baby_gpt_instruct.pth"
INSTRUCT_CKPT  = f"{DRIVE_DIR}/instruct_checkpoint.pth"
BACKUP_DIR     = f"{DRIVE_DIR}/backup"                    


# ============================================================
# FINE-TUNE HYPERPARAMS
# ============================================================

FINETUNE_LR    = 1e-5        # 30x smaller than base (3e-4)
FINETUNE_ITERS = 3000        # ~4.5 epochs over 21k pairs
WARMUP_STEPS   = 100         # linear warmup
CKPT_EVERY     = 250         # save every 250 steps
EVAL_EVERY     = 100
EVAL_ITERS     = 30
GRAD_CLIP      = 1.0


# ============================================================
# HELPERS
# ============================================================

def safe_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def backup(src, tag):
    """Save a timestamped copy of a checkpoint locally (in Google Drive)."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dst = os.path.join(BACKUP_DIR, f"{tag}.pth")
    shutil.copy2(src, dst)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Device: {device}")
if torch.cuda.is_available():
    print(f"🎮 {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
else:
    print("⚠️  No GPU detected — training will be slow on CPU.")


# ============================================================
# LOAD FINE-TUNE DATA
# ============================================================

if not os.path.exists(FINETUNE_BIN):
    raise FileNotFoundError(
        f"❌ {FINETUNE_BIN} not found. Make sure it is in your Google Drive folder."
    )

data = np.memmap(FINETUNE_BIN, dtype=np.uint16, mode="r")
print(f"📚 Fine-tune tokens: {len(data):,}")

n = int(0.95 * len(data))     # 95/5 split
train_data = data[:n]
val_data   = data[n:]
print(f"📊 Train: {len(train_data):,}   Val: {len(val_data):,}")


def get_batch(split):
    ds = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(ds) - block_size, size=batch_size)
    x = np.stack([ds[i:i + block_size] for i in ix])
    y = np.stack([ds[i + 1:i + block_size + 1] for i in ix])
    x = torch.from_numpy(x.astype(np.int64))
    y = torch.from_numpy(y.astype(np.int64))
    return x.to(device), y.to(device)


# ============================================================
# LOAD BASE MODEL
# ============================================================

if not os.path.exists(BASE_CKPT):
    raise FileNotFoundError(f"❌ Base checkpoint not found: {BASE_CKPT}")

print(f"\n🧠 Loading base: {BASE_CKPT}")
model = BabyGPT().to(device)

# FIXED: Added weights_only=False for PyTorch 2.6+
base = torch.load(BASE_CKPT, map_location=device, weights_only=False)

# Handle if the checkpoint is wrapped in a dict
if "model_state_dict" in base:
    model.load_state_dict(base["model_state_dict"])
    print(f"✅ Base loaded from step {base.get('step', 'unknown')} "
          f"(loss {base.get('loss', 0):.4f})")
else:
    model.load_state_dict(base)
    print("✅ Base loaded successfully.")

total_params = sum(p.numel() for p in model.parameters())
print(f"   Params: {total_params:,}")

# Fresh optimizer (don't reuse base's AdamW state)
optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR)


# ============================================================
# OPTIONAL: RESUME FINE-TUNE
# ============================================================

start_iter = 0
if os.path.exists(INSTRUCT_CKPT):
    print(f"\n📂 Resuming fine-tune from {INSTRUCT_CKPT}")
    try:
        # FIXED: Added weights_only=False here too
        ic = torch.load(INSTRUCT_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ic["model_state_dict"])
        optimizer.load_state_dict(ic["optimizer_state_dict"])
        start_iter = ic["step"] + 1
        print(f"✅ Resumed at step {start_iter}")
    except Exception as e:
        print(f"⚠️ Could not resume from checkpoint ({e}).")
        print("🔄 Starting fresh from step 0 instead!")


# ============================================================
# LR SCHEDULE (warmup + cosine)
# ============================================================

def lr_at(step):
    if step < WARMUP_STEPS:
        return FINETUNE_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, FINETUNE_ITERS - WARMUP_STEPS)
    progress = min(1.0, progress)
    return 0.1 * FINETUNE_LR + 0.9 * FINETUNE_LR * 0.5 * (1 + np.cos(np.pi * progress))


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def estimate_loss(iters=EVAL_ITERS):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(iters)
        for k in range(iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ============================================================
# TRAIN LOOP
# ============================================================

print("\n" + "=" * 60)
print(f"🏋️  FINE-TUNING  (LR={FINETUNE_LR}, iters={FINETUNE_ITERS})")
print("=" * 60)

for step in range(start_iter, FINETUNE_ITERS):

    lr = lr_at(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        print(f"Step {step:5d} | lr {lr:.2e} | "
              f"Train {losses['train']:.4f} | Val {losses['val']:.4f}")

    if step % CKPT_EVERY == 0 and step > 0:
        ckpt_obj = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }
        safe_save(ckpt_obj, INSTRUCT_CKPT)
        backup(INSTRUCT_CKPT, f"instruct_step_{step}")
        print(f"💾 Step {step} → saved to Google Drive + backup")


# ============================================================
# FINAL SAVE
# ============================================================

print("\n" + "=" * 60)
print("💾 SAVING INSTRUCT MODEL")
print("=" * 60)

safe_save(model.state_dict(), INSTRUCT_OUT)
print(f"✅ Saved to: {INSTRUCT_OUT}")

print("\n🎉 FINE-TUNING COMPLETE")