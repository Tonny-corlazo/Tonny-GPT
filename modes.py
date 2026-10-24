import torch
import os
from tokenizer import GPTTokenizer
from model import BabyGPT

print("=" * 60)
print("🍼 BABYGPT PERSONALITY LAB")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Using device: {device}\n")

# ============================================================
# LOAD TOKENIZER + MODEL
# ============================================================

tokenizer = GPTTokenizer()
model = BabyGPT()
model.to(device)

final_model_path = "baby_gpt_mega.pth"
checkpoint_path = "checkpoint.pth"

if os.path.exists(final_model_path):

    print(f"📂 Using final model: {final_model_path}")

    state = torch.load(
        final_model_path,
        map_location=device
    )

    if "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)

    source = "final"

elif os.path.exists(checkpoint_path):

    print(f"📂 Using checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(f"   Step: {checkpoint.get('step')}")
    print(f"   Loss: {checkpoint.get('loss'):.4f}")

    source = f"checkpoint@{checkpoint.get('step')}"

else:
    raise FileNotFoundError(
        "No model weights found."
    )

model.eval()

# ============================================================
# PERSONALITIES
# ============================================================

personalities = {

    "1": {
        "name": "💬 Chat",
        "instruction": """
You are BabyGPT Chat.

You are a casual, friendly conversational assistant.
Talk naturally and keep responses conversational.
Do not over-explain unless asked.
"""
    },

    "2": {
        "name": "❓ QA",
        "instruction": """
You are BabyGPT QA.

Your job is to answer questions directly.
Give the most relevant answer you can.
Do not unnecessarily change the subject.
If you don't know something, say so rather than inventing
an unrelated answer.
"""
    },

    "3": {
        "name": "🔬 Science",
        "instruction": """
You are BabyGPT Science.

You specialize in science.
Explain scientific concepts clearly and accurately.
Use simple examples when useful.
Stay focused on the scientific question.
"""
    },

    "4": {
        "name": "📜 History",
        "instruction": """
You are BabyGPT History.

You specialize in history.
Focus on historical events, people, dates, causes,
and consequences.
Explain events in a clear chronological way when useful.
"""
    },

    "5": {
        "name": "💻 Coding",
        "instruction": """
You are BabyGPT Code.

You specialize in programming and computer science.
Focus on the programming question.
When giving code, explain what the important parts do.
"""
    }
}


# ============================================================
# CHOOSE PERSONALITY
# ============================================================

while True:

    print("\n" + "=" * 60)
    print("🎭 CHOOSE BABYGPT PERSONALITY")
    print("=" * 60)

    for key, personality in personalities.items():
        print(f"{key}. {personality['name']}")

    print("0. 🚪 Exit")

    choice = input("\nChoose personality: ").strip()

    if choice == "0":
        print("\n👋 BabyGPT going back to the nursery.")
        break

    if choice not in personalities:
        print("❌ Invalid choice.")
        continue

    personality = personalities[choice]

    print("\n" + "=" * 60)
    print(f"🎭 PERSONALITY: {personality['name']}")
    print("=" * 60)

    print("\nType 'back' to choose another personality.")
    print("Type 'exit' to quit.\n")

    # ========================================================
    # CHAT LOOP
    # ========================================================

    while True:

        question = input("Human: ").strip()

        if question.lower() == "back":
            break

        if question.lower() == "exit":
            print("\n👋 BabyGPT shutting down.")
            raise SystemExit

        prompt = (
            personality["instruction"]
            + "\n\n"
            + "Human: "
            + question
            + "\nAssistant:"
        )

        token_ids = tokenizer.encode(prompt)

        context = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():

            _, generated_tokens = model.generate(
                context,
                max_new_tokens=100,
                temperature=0.8,
                top_k=40,
                top_p=None,
                stop_tokens=[
                    13,
                    628,
                    198,
                    50256
                ]
            )

        answer = tokenizer.decode(generated_tokens)

        print(f"\nAssistant: {answer}\n")