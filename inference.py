# inference.py
import torch
from model import BabyGPT
from tokenizer import GPTTokenizer

def load_model(model_path="baby_gpt_instruct_final.pth"):
    """Load the trained model safely"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BabyGPT()
    # FIX 1: weights_only=False and handle the dictionary format
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    return model, device

def chat_loop():
    """Interactive chat loop — like ChatGPT!"""
    print("=" * 60)
    print("🤖 TONY-GPT CHAT")
    print("=" * 60)
    print("Type 'exit' to quit, 'clear' to clear history")
    print("-" * 60)
    
    # Load model and device
    model, device = load_model()
    tokenizer = GPTTokenizer()
    
    history = ""
    
    while True:
        user_input = input("\n🧑 You: ")
        
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break
        elif user_input.lower() == "clear":
            history = ""
            print("🧹 History cleared!")
            continue
        
        # Add user input to history
        history += f"Human: {user_input}\nAssistant:"
        
        print("🤖 Thinking...", end="", flush=True)
        
        # FIX 2: Generate directly instead of using model.chat() to avoid list errors
        ids = tokenizer.encode(history)
        context = torch.tensor([ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            # Generate the response
            generated = model.generate(
                context, 
                max_new_tokens=100, 
                temperature=0.3, # Lower temp to prevent repetition loops
                top_k=40
            )[0].tolist()
            
        # Decode the output
        decoded_text = tokenizer.decode(generated)
        if isinstance(decoded_text, list):
            decoded_text = decoded_text[0]
            
        # Extract ONLY the new assistant response
        response = decoded_text[len(history):].strip()
        
        # Cut off if it tries to hallucinate the next "Human:" turn
        response = response.split("Human:")[0].strip()
        
        print(f"\r🤖: {response}")
        
        # Add the AI's response to history so it remembers context for the next turn
        history += f" {response}\n"

if __name__ == "__main__":
    chat_loop()