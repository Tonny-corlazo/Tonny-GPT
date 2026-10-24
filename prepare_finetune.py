# prepare_finetune.py
import os
import re
import json
from datasets import load_dataset

DRIVE_DIR = "/content/drive/MyDrive/Babay GPT/Datasets/conversations"
WORKDIR = "/content/drive/MyDrive/Babay GPT"
OUT_JSONL = f"{WORKDIR}/finetune_data.jsonl"

MAX_PAIRS = 30000
MIN_RESP, MIN_PROMPT, MAX_PROMPT, MAX_RESP = 20, 3, 800, 2000

CONVO_FILES = [
    f"{DRIVE_DIR}/ultrachat_conversations.txt",
    f"{DRIVE_DIR}/sharegpt_conversations.txt",
    f"{DRIVE_DIR}/openhermes_conversations.txt",
    f"{DRIVE_DIR}/chatgpt_1m_conversations.txt",
]

def is_clean(text):
    if not text or not text.strip(): return False
    if re.search(r"(.)\1{6,}", text): return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if len(text) > 0 and non_ascii / len(text) > 0.3: return False
    return True

def validate_pair(p, r):
    return (MIN_PROMPT <= len(p) <= MAX_PROMPT) and (MIN_RESP <= len(r) <= MAX_RESP) and is_clean(p) and is_clean(r)

# --- Regex Formats ---
RE_HUMAN = re.compile(r"Human:\s*(.*?)\s*Assistant:\s*(.*?)(?=Human:|\Z)", re.DOTALL | re.IGNORECASE)
RE_HASH = re.compile(r"###\s*Human:\s*(.*?)\s*###\s*Assistant:\s*(.*?)(?=###\s*Human:|\Z)", re.DOTALL | re.IGNORECASE)
RE_CHATML = re.compile(r"<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>\s*<\|im_start\|>assistant\s*(.*?)\s*<\|im_end\|>", re.DOTALL | re.IGNORECASE)
RE_PIPE = re.compile(r"<\|user\|>\s*(.*?)\s*<\|assistant\|>\s*(.*?)(?=<\|user\|>|\Z)", re.DOTALL | re.IGNORECASE)
FORMATS = [RE_HUMAN, RE_HASH, RE_CHATML, RE_PIPE]

def extract_pairs(text):
    best_pairs = []
    for regex in FORMATS:
        pairs = []
        for m in regex.finditer(text):
            p, r = m.group(1).strip(), m.group(2).strip()
            if validate_pair(p, r):
                pairs.append((p, r))
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
    return best_pairs

# --- 1. EXTRACT LOCAL FILES ---
print(" Extracting from local files...")
all_pairs = []
seen = set()

for path in CONVO_FILES:
    if not os.path.exists(path):
        print(f"⚠️ Missing: {path}")
        continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    
    pairs = extract_pairs(raw)
    print(f"  Found {len(pairs):,} pairs in {os.path.basename(path)}")
    
    for p, r in pairs:
        # FIXED: Deduplicate using the full prompt, not just 100 chars
        key = p.lower()
        if key not in seen:
            seen.add(key)
            all_pairs.append({"prompt": p, "response": r})
        if len(all_pairs) >= MAX_PAIRS: break

# --- 2. ADD DOLLY WITH ERROR HANDLING ---
print(" Adding Dolly-15k...")
try:
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly_count = 0
    for item in dolly:
        if len(all_pairs) >= MAX_PAIRS: break
        prompt = f"{item.get('context', '')} {item['instruction']}".strip()
        response = item["response"]
        
        if validate_pair(prompt, response):
            key = prompt.lower()
            if key not in seen:
                seen.add(key)
                all_pairs.append({"prompt": prompt, "response": response})
                dolly_count += 1
    print(f"  Added {dolly_count:,} Dolly pairs.")
except Exception as e:
    print(f"⚠️ Could not load Dolly: {e}")

# --- 3. SAVE JSONL ---
print(f"💾 Saving {len(all_pairs):,} total pairs to {OUT_JSONL}...")
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for pair in all_pairs:
        f.write(json.dumps(pair) + "\n")

print("✅ Preparation Complete!")
