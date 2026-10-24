# 🧠 Tony-GPT (BabyGPT)

A fully functional, 89-Million parameter Large Language Model built completely from scratch using PyTorch and the mercies of gods..HA!HA!HA!

* -- Does it Hallucinate, Only one way to find out. 

## 🚀 The Architecture
Tony-GPT is not a wrapper around an API. It is a custom Transformer built from the ground up, featuring:
*   **Multi-Head Self-Attention** (12 layers, 8 heads)
*   **Custom GPT-2 Tokenizer** integration
*   **Masked Causal Language Modeling** (Loss masking on the Assistant's responses only)
*   **Repetition Penalty** generation logic to prevent infinite loops.

## 🏋️ The Training Pipeline
1.  **Pre-Training (Base Model):** Trained on raw text (Wikipedia, books, etc.) to learn English grammar and facts.
2.  **Instruction Fine-Tuning (Instruct Model):** Trained on 30,000+ prompt-response pairs (Databricks Dolly-15k + custom conversational data) to learn how to be an assistant. 

## 📂 Project Structure
*   `model.py`: The Transformer architecture (Attention, FeedForward, LayerNorm).
*   `tokenizer.py`: Custom wrapper for the GPT-2 tokenizer.
*   `prepare_finetune.py`: Data pipeline to clean and format JSONL conversation data.
*   `finetune.py`: The training loop with proper loss masking (`-100` for Human tokens).
*   `inference.py`: Interactive CLI chat interface.

## 💻 How to Run
*(Note: Model weights `.pth` are excluded from this repo due to size)*
```bash
pip install -r requirements.txt
python inference.py