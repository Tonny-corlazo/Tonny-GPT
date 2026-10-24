import torch
from config import block_size
from model import BabyGPT
from tokenizer import GPTTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BabyGPT().to(device)

# Point this to the model that finished training
model_path = "./baby_gpt_instruct_final.pth" # Make sure this matches your file name!
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
    
model.eval()
tokenizer = GPTTokenizer()

prompts = [
    "Human: Hey, how are you?\nAssistant:",
    "Human: What is 2 + 2?\nAssistant:",
    "Human: Tell me a joke.\nAssistant:",
    "Human: Who are you?\nAssistant:",
    "Human: What is the capital of France?\nAssistant:"
]

print("🤖 TESTING TONY-GPT...\n" + "="*40)

for p in prompts:
    ids = tokenizer.encode(p)
    context = torch.tensor([ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=50, temperature=0.7, top_k=40)[0].tolist()
        
    # --- THE FIX ---
    decoded_text = tokenizer.decode(generated)
    
    if isinstance(decoded_text, list):
        decoded_text = decoded_text[0]
        
    answer = decoded_text[len(p):].strip()
    answer = answer.split("Human:")[0].strip()
    
    print(f"Q: {p.splitlines()[0].replace('Human: ', '')}")
    print(f"A: {answer}\n")