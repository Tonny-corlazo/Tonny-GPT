# preprocess.py

import os
import numpy as np
from tokenizer import GPTTokenizer

CORPUS_PATH = "training_corpus.txt"
OUTPUT_PATH = "encoded_corpus.bin"

CHUNK_SIZE = 1_000_000  # 1 million characters


print("=" * 60)
print("🔢 BABYGPT CORPUS TOKENIZATION")
print("=" * 60)

print("\n📂 Loading GPT-2 tokenizer...")
tokenizer = GPTTokenizer()

print("✅ Tokenizer ready")

# --------------------------------------------------
# Resume support
# --------------------------------------------------

existing_tokens = 0

if os.path.exists(OUTPUT_PATH):
    existing_bytes = os.path.getsize(OUTPUT_PATH)
    existing_tokens = existing_bytes // 2

    print(
        f"\n📂 Existing encoded corpus found!"
        f"\n   Tokens already written: {existing_tokens:,}"
    )

    # For a simple first run, start fresh.
    # We'll make proper resumable preprocessing below.
    os.remove(OUTPUT_PATH)
    existing_tokens = 0


print("\n📚 Encoding corpus in chunks...")

total_chars = 0
total_tokens = 0
chunk_number = 0

with open(CORPUS_PATH, "r", encoding="utf-8") as source, \
     open(OUTPUT_PATH, "wb") as output:

    while True:

        text_chunk = source.read(CHUNK_SIZE)

        if not text_chunk:
            break

        chunk_number += 1
        total_chars += len(text_chunk)

        # Tokenize ONLY this chunk
        tokens = tokenizer.encode(text_chunk)

        # GPT-2 vocabulary = 50,257
        # uint16 can hold up to 65,535
        token_array = np.asarray(tokens, dtype=np.uint16)

        output.write(token_array.tobytes())

        total_tokens += len(token_array)

        print(
            f"\r🧩 Chunk {chunk_number:,} | "
            f"Characters: {total_chars:,} | "
            f"Tokens: {total_tokens:,}",
            end=""
        )

print("\n")
print("=" * 60)
print("✅ TOKENIZATION COMPLETE")
print("=" * 60)

print(f"Characters: {total_chars:,}")
print(f"Tokens:     {total_tokens:,}")

file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

print(f"Binary size: {file_size_mb:.2f} MB")
print(f"Saved to:   {OUTPUT_PATH}")