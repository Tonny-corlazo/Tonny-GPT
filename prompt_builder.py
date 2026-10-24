# ptompt_builder.py

import torch
import os
from config import block_size
from tokenizer import BPETokenizer
from model import BabyGPT

print("=" * 60)
print("🤖 BABYGPT MEGA — CHAT MODE")
print("=" * 60)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Using device: {device}")

print("\n📂 Loading tokenizer...")
tokenizer = BPETokenizer()
tokenizer.load("tokenizer_mega_10k.json")
vocab_size = tokenizer.next_token
print(f"✅ Tokenizer loaded! Vocab size: {vocab_size}")

# 2. Load model
print("\n🧠 Loading model...")
model = BabyGPT()
model.to(device)

model_path = "baby_gpt_mega.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Model loaded from '{model_path}'")
else:
    print(f"❌ Model file '{model_path}' not found!")
    exit()

model.eval()
print("✅ Model ready!")

# ============================================
# 🔥 SYSTEM PROMPT — !
# ============================================

System_prompt = """You are a helpful AI Assistant named Lyra, you are trained on a diverse corpus including
-The Bible
-Alice in Wonderland
-Freud's Interpretation of Dreams
-Darwin's Origin of Species
-Marcus Aurelius' Meditations
-JavaScript
-Physics (Hawking)
-How to Winf Friends and Influence People


You answer questions clearly, conscisely, and thoughtfully.
You can explain concepts, write code, or have philosophical conversations.
You always respond in a friendly, helpful tone
"""

def ask_model(question, max_new_tokens=300, temperature=0.7):
    """Ask the model a question with the system prompt."""

    #Buid the full prompt
    full_prompt = f"{System_prompt}\n\nUser: {question}\n\nAssistant:"

    #Encode
    context = torch.tensor([tokenizer.encode(full_prompt)]).to(device)

    #generate
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            top_k=40,
            top_p=0.9
        )

        #Decode
        full_output = tokenizer.decode(generated[0].tolist())

        #Extract only the assistants response
        assistant_start = "Assistant:"
        if assistant_start in full_output:
            response = full_output.split(assistant_start)[-1].strip()
        else:
            response = full_output
        
        return response
    
 # ============================================
# 🎯 CHAT LOOP
# ============================================
print("\n" + "=" * 60)
print("💬 CHAT MODE — Type 'exit' to quit")
print("=" * 60)

while True:
    user_input = input("\n🧑 You: ").strip()

    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("\n👋 Goodbye, Boss! Keep building! 🚀")
        break

    if not user_input:
        continue

    print("\n🤖 Thinking...\n")

    response = ask_model(
        question=user_input,
        max_new_tokens=500,
        temperature=0.7
    )

# ============================================
# 🧪 SINGLE QUERY MODE (if you prefer one-off)
# ============================================

def ask_once(question):
    """Ask a single question without the chat loop."""
    response = ask_model(question, max_new_tokens=500, temperature=0.7)
    print(f"\n🤖 {response}")
    return response

if __name__ == "__main__":
    # Test single query
    test_question = "What is the meaning of life?"
    print(f"\n🧪 Test: {test_question}")
    response = ask_model(test_question, max_new_tokens=100, temperature=0.7)
    print(f"🤖 {response}")

