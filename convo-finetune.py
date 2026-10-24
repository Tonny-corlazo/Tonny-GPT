# finetune.py — The CTO-Approved Version
import os, json, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizer import GPTTokenizer
from model import BabyGPT
from config import block_size

DRIVE_DIR = "/content/drive/MyDrive/Babay GPT"
BASE_CKPT = f"{DRIVE_DIR}/checkpoint.pth"
DATA_JSONL = f"{DRIVE_DIR}/finetune_data.jsonl"
INSTRUCT_OUT = f"{DRIVE_DIR}/baby_gpt_instruct_final.pth"
CKPT_PATH = f"{DRIVE_DIR}/instruct_ckpt_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPTTokenizer()

# Verify tokenizer has EOS before doing anything
if not hasattr(tokenizer, "eos_token_id"):
    raise AttributeError("❌ Tokenizer missing 'eos_token_id'. Check tokenizer.py!")
EOS_ID = tokenizer.eos_token_id

class InstructDataset(Dataset):
    def __init__(self, jsonl_path, max_length=block_size):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        if len(self.data) == 0:
            raise ValueError("❌ Dataset is empty! Check prepare_finetune.py.")
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_ids = tokenizer.encode(f"Human: {item['prompt']}\nAssistant: ")
        response_ids = tokenizer.encode(item['response']) + [EOS_ID]
        
        full_ids = prompt_ids + response_ids
        
        # Handle Truncation
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            
        # Handle Padding
        padding_length = self.max_length - len(full_ids)
        if padding_length > 0:
            full_ids = full_ids + [EOS_ID] * padding_length
        
        x = torch.tensor(full_ids[:-1], dtype=torch.long)
        y = torch.tensor(full_ids[1:], dtype=torch.long)
        
        # --- THE PERFECT MASKING LOGIC ---
        start_of_response = len(prompt_ids) - 1
        
        # 1. Mask the prompt (Human)
        if start_of_response > 0:
            y[:start_of_response] = -100
            
        # 2. Mask ONLY the padding, preserve the EOS of the response!
        if padding_length > 0:
            # The padding starts after the prompt + response - 1 (for the shift)
            pad_start_idx = len(prompt_ids) + len(response_ids) - 1
            if pad_start_idx < self.max_length:
                y[pad_start_idx:] = -100
        
        return x, y

# --- SETUP ---
print("📂 Loading data...")
dataset = InstructDataset(DATA_JSONL)
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Verify splits aren't empty
if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("❌ Train or Validation split is empty. Need more data!")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print(f"🧠 Loading base model: {BASE_CKPT}")
model = BabyGPT().to(device)
base = torch.load(BASE_CKPT, map_location=device, weights_only=False)
model.load_state_dict(base["model_state_dict"] if "model_state_dict" in base else base)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# --- RESUME LOGIC ---
start_epoch = 0
if os.path.exists(CKPT_PATH):
    print(f"📂 Resuming from checkpoint: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1

# --- TRAINING LOOP ---
EPOCHS = 3
print("🏋️ Starting CTO-Approved Training...")
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_train_loss = 0
    
    for step, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
            
        if step % 100 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Train Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            if loss is not None:
                total_val_loss += loss.item()
                
    avg_train = total_train_loss / len(train_loader)
    avg_val = total_val_loss / len(val_loader)
    print(f"✅ Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
    
    # --- SAVE RESUMABLE CHECKPOINT ---
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train,
        "val_loss": avg_val
    }
    torch.save(checkpoint, CKPT_PATH)
    print(f"💾 Saved resumable checkpoint after epoch {epoch+1}")

# Final inference save (no optimizer state)
torch.save(model.state_dict(), INSTRUCT_OUT)
print(f"✅ Training Complete. Final model saved to {INSTRUCT_OUT}")