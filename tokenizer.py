from transformers import GPT2Tokenizer

class GPTTokenizer:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.vocab_size = self.tokenizer.vocab_size
        print(f"Loaded GPT-2 tokenizer: {self.vocab_size:,} tokens; EOS/PAD={self.pad_token_id}")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts):
        return self.tokenizer(texts, padding=True, return_tensors="pt",
                              add_special_tokens=False)["input_ids"]

    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        return self.vocab_size
