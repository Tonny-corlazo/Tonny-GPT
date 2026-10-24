# generate.py
import torch
import os
from tokenizer import GPTTokenizer
from model import BabyGPT

print("=" * 60)
print("🎨 EVALUATING BABYGPT")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Using device: {device}\n")

tokenizer = GPTTokenizer()
model = BabyGPT()
model.to(device)

# Load final model if present, else checkpoint
final_model_path = "baby_gpt_mega.pth"
checkpoint_path = "instruct_step_2000.pth"

if os.path.exists(final_model_path):
    print(f"📂 Using final model: {final_model_path}")
    state = torch.load(final_model_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    source = "final"

elif os.path.exists(checkpoint_path):
    print(f"📂 Using checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"   Step: {checkpoint.get('step')}")
    print(f"   Loss: {checkpoint.get('loss'):.4f}")
    source = f"checkpoint@{checkpoint.get('step')}"

else:
    raise FileNotFoundError("No model weights found.")

model.eval()

# ============================================================
# EVALUATION SET
# ============================================================
prompts = [
    # ===== CASUAL CONVERSATION =====
    "Human: Hey, how are you?\nAssistant:",
    "Human: What's up?\nAssistant:",
    "Human: How was your day?\nAssistant:",
    "Human: I'm bored asf 😂\nAssistant:",
    "Human: What are you doing right now?\nAssistant:",
    "Human: Tell me something interesting.\nAssistant:",
    "Human: I just got home.\nAssistant:",
    "Human: It's been a long day.\nAssistant:",

    # ===== BANTER / HUMOR =====
    "Human: Bro, you're funny 😂\nAssistant:",
    "Human: You really think you're smart huh? 😂\nAssistant:",
    "Human: Stop acting like you know everything 😂\nAssistant:",
    "Human: I can't believe you actually did that.\nAssistant:",
    "Human: You're crazy 😂\nAssistant:",
    "Human: Why are you like this? 😂\nAssistant:",
    "Human: Be honest with me.\nAssistant:",
    "Human: Tell me a joke.\nAssistant:",

    # ===== PERSONAL / EMOTIONAL =====
    "Human: I'm feeling really happy today.\nAssistant:",
    "Human: I'm having a terrible day.\nAssistant:",
    "Human: I passed my exam!\nAssistant:",
    "Human: I failed my exam and I'm disappointed.\nAssistant:",
    "Human: I think I have a crush on someone.\nAssistant:",
    "Human: I don't know what to do with my life right now.\nAssistant:",
    "Human: I feel lonely today.\nAssistant:",
    "Human: I just need someone to talk to.\nAssistant:",
    "Human: I'm stressed about school.\nAssistant:",
    "Human: I have something bothering me.\nAssistant:",

    # ===== OPINIONS / OPEN-ENDED =====
    "Human: What do you think makes someone happy?\nAssistant:",
    "Human: What makes a good friend?\nAssistant:",
    "Human: Do you think people change?\nAssistant:",
    "Human: What's something everyone should experience?\nAssistant:",
    "Human: What's your favorite kind of conversation?\nAssistant:",
    "Human: What would you do if you were me?\nAssistant:",
    "Human: Give me some advice.\nAssistant:",
    "Human: What do you think about relationships?\nAssistant:",

    # ===== SOCIAL SITUATIONS =====
    "Human: I met someone interesting today.\nAssistant:",
    "Human: Someone gave me their number today 😂\nAssistant:",
    "Human: I don't know if this person likes me.\nAssistant:",
    "Human: My friend hasn't replied to my message.\nAssistant:",
    "Human: I want to talk to someone but I don't know what to say.\nAssistant:",
    "Human: Someone keeps teasing me 😂\nAssistant:",
    "Human: I think someone is avoiding me.\nAssistant:",
    "Human: My friend just told me some crazy news.\nAssistant:",

    # ===== ENTERTAINMENT =====
    "Human: I don't know what movie to watch.\nAssistant:",
    "Human: I want something scary to watch tonight.\nAssistant:",
    "Human: Recommend me something interesting.\nAssistant:",
    "Human: I just watched a really good movie.\nAssistant:",
    "Human: I love horror movies.\nAssistant:",
    "Human: Let's talk about movies.\nAssistant:",
    "Human: What's the weirdest movie you've seen?\nAssistant:",

    # ===== RANDOM / NATURAL CHAT =====
    "Human: Guess what happened today?\nAssistant:",
    "Human: You won't believe this.\nAssistant:",
    "Human: Wait, listen to this 😂\nAssistant:",
    "Human: I have a funny story.\nAssistant:",
    "Human: Okay, hear me out.\nAssistant:",
    "Human: Don't judge me 😂\nAssistant:",
    "Human: I have a crazy idea.\nAssistant:",
    "Human: Wanna hear something random?\nAssistant:",

    # ===== MULTI-TURN STYLE =====
    "Human: Hey, I'm bored.\nAssistant:",
    "Human: Yeah, I don't know what to do today.\nAssistant:",
    "Human: Maybe I should watch a movie.\nAssistant:",
    "Human: I like horror movies.\nAssistant:",
    "Human: I watched Let Me In yesterday 😂\nAssistant:",
    "Human: That movie was actually disturbing.\nAssistant:",
    "Human: Now I need something else to watch.\nAssistant:",
    "Human: What should I do with the rest of my Sunday?\nAssistant:",
]

print("=" * 60)
print(f"📝 EVALUATION — source: {source}")
print("=" * 60)

for i, prompt in enumerate(prompts, 1):
    token_ids = tokenizer.encode(prompt)
    context = torch.tensor([token_ids], dtype=torch.long).to(device)

    _, generated_tokens = model.generate(
        context,
        max_new_tokens=100,
        temperature=0.7,
        top_k=40,
        top_p=None,
        stop_tokens=[198, 50256]
    )

    answer = tokenizer.decode(generated_tokens)

    print(f"\n[{i}] Q: {prompt.splitlines()[0].replace('Human: ', '')}")
    print(f"    A: {answer}")