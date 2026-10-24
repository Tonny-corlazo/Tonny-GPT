# process_massive_datasets.py
from datasets import load_dataset
import os
import time


# ============================================================
# TEXT DATASET PROCESSOR
# ============================================================
def process_dataset_streaming(dataset, name, max_samples=10000,
                              min_chars=100, output_file=None):
    if output_file is None:
        output_file = f"{name.lower().replace('/', '_')}_corpus.txt"

    print(f"\n📚 Processing {name}...")
    print(f"📁 Saving to: {output_file}")

    start_time = time.time()
    count = 0

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break

                text = None
                for key in ["text", "content", "article", "paragraph", "body", "abstract"]:
                    if key in item and isinstance(item[key], str):
                        text = item[key]
                        break

                if text and len(text) > min_chars:
                    f.write(text + "\n\n")
                    count += 1

                    if count % 1000 == 0:
                        elapsed = time.time() - start_time
                        print(f"  {count} samples | {elapsed:.1f}s")

        elapsed = time.time() - start_time
        print(f"✅ {name}: {count} samples in {elapsed:.1f}s")

        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"📊 File size: {size / 1024 / 1024:.2f} MB")

        return count

    except Exception as e:
        print(f"⚠️ Error processing {name}: {e}")
        return 0


# ============================================================
# CONVERSATION DATASET PROCESSOR
# ============================================================
def process_conversations(dataset, name, max_samples=100000,
                          target_chars=20_000_000, output_file=None,
                          min_text=20, max_text=10000):
    if output_file is None:
        output_file = f"{name.lower().replace('/', '_')}_conversations.txt"

    print(f"\n💬 Processing {name}...")
    print(f"📁 Saving to: {output_file}")
    print(f"🎯 Target: {target_chars / 1_000_000:.0f}M chars")

    start_time = time.time()
    convos_written = 0
    chars_written = 0

    def extract_turns(item):
        """Return a list of {role, text} turns from various schemas."""
        raw = (
            item.get("conversations")
            or item.get("messages")
            or item.get("conversation")
            or []
        )

        # Some datasets have turns as a JSON string — try to decode
        if isinstance(raw, str):
            import json
            try:
                raw = json.loads(raw)
            except Exception:
                return []

        turns = []
        for turn in raw:
            if not isinstance(turn, dict):
                continue

            role = (
                turn.get("role")
                or turn.get("from")
                or turn.get("speaker")
                or ""
            ).lower()

            text = (
                turn.get("content")
                or turn.get("value")
                or turn.get("text")
                or ""
            )
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue

            # Normalize
            if role in ("system",):
                # Skip system prompts
                continue
            if role in ("user", "human", "prompter"):
                turns.append(("Human", text))
            elif role in ("assistant", "gpt", "bot", "ai"):
                turns.append(("Assistant", text))

        return turns

    def format_conversation(item):
        turns = extract_turns(item)
        if len(turns) < 2:
            return None
        return "\n".join(f"{role}: {text}" for role, text in turns)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            inspected = 0
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break

                inspected += 1

                # Print progress every 200 items, regardless of filtering
                if inspected % 200 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"  inspected {inspected} | "
                        f"kept {convos_written} | "
                        f"{chars_written/1_000_000:.1f}M chars | "
                        f"{elapsed:.1f}s"
                    )

                text = format_conversation(item)
                if not text:
                    continue
                if len(text) < min_text or len(text) > max_text:
                    continue

                f.write(text + "\n\n")
                convos_written += 1
                chars_written += len(text)

                if chars_written >= target_chars:
                    print(f"🎯 Hit target at {chars_written:,} chars")
                    break

        elapsed = time.time() - start_time
        print(f"✅ {name}: {convos_written} convos from {inspected} items in {elapsed:.1f}s")

        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"📊 File size: {size / 1024 / 1024:.2f} MB")

        return convos_written

    except Exception as e:
        print(f"⚠️ Error processing {name}: {e}")
        return 0


# ============================================================
# KNOWLEDGE DATASETS
# ============================================================
def process_c4():
    print("\n" + "=" * 60)
    print("🌐 PROCESSING C4")
    print("=" * 60)
    try:
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        return process_dataset_streaming(dataset, "C4", max_samples=50000)
    except Exception as e:
        print(f"⚠️ C4 error: {e}")
        return 0


def process_the_pile():
    print("\n" + "=" * 60)
    print("📚 PROCESSING THE PILE")
    print("=" * 60)
    try:
        dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        return process_dataset_streaming(dataset, "The_Pile", max_samples=20000)
    except Exception as e:
        print(f"⚠️ The Pile error: {e}")
        return 0


def process_openwebmath():
    print("\n" + "=" * 60)
    print("📐 PROCESSING OPENWEBMATH")
    print("=" * 60)
    try:
        dataset = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        return process_dataset_streaming(dataset, "OpenWebMath", max_samples=20000)
    except Exception as e:
        print(f"⚠️ OpenWebMath error: {e}")
        return 0


def process_wikipedia():
    print("\n" + "=" * 60)
    print("🌐 PROCESSING WIKIPEDIA")
    print("=" * 60)
    try:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        return process_dataset_streaming(
            dataset, "Wikipedia", max_samples=20000, output_file="wikipedia_corpus.txt"
        )
    except Exception as e:
        print(f"⚠️ Wikipedia error: {e}")
        return 0


# ============================================================
# CONVERSATION DATASETS
# ============================================================
def process_ultrachat():
    """UltraChat — clean, consistent schema, fast."""
    print("\n" + "=" * 60)
    print("💬 PROCESSING ULTRACHAT 200K")
    print("=" * 60)
    try:
        dataset = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            streaming=True,
        )
        return process_conversations(
            dataset, "UltraChat", max_samples=100000, target_chars=20_000_000
        )
    except Exception as e:
        print(f"⚠️ UltraChat error: {e}")
        return 0


def process_sharegpt():
    print("\n" + "=" * 60)
    print("💬 PROCESSING SHAREGPT")
    print("=" * 60)
    try:
        dataset = load_dataset(
            "Aeala/ShareGPT_Vicuna_unfiltered",
            split="train",
            streaming=True,
        )
        return process_conversations(
            dataset, "ShareGPT", max_samples=100000, target_chars=20_000_000
        )
    except Exception as e:
        print(f"⚠️ ShareGPT error: {e}")
        return 0


def process_openhermes():
    print("\n" + "=" * 60)
    print("💬 PROCESSING OPENHERMES 2.5")
    print("=" * 60)
    try:
        dataset = load_dataset(
            "teknium/OpenHermes-2.5",
            split="train",
            streaming=True,
        )
        return process_conversations(
            dataset, "OpenHermes", max_samples=100000, target_chars=15_000_000
        )
    except Exception as e:
        print(f"⚠️ OpenHermes error: {e}")
        return 0


def process_chatgpt_multi_turn():
    print("\n" + "=" * 60)
    print("💬 PROCESSING CHATGPT-STYLE 1M MULTI-TURN")
    print("=" * 60)
    try:
        dataset = load_dataset(
            "Gopinath-AI/ChatGPT-Style_1M_Conversation_Dataset_Multi-Turn",
            split="train",
            streaming=True,
        )
        return process_conversations(
            dataset, "ChatGPT_1M", max_samples=200000, target_chars=20_000_000
        )
    except Exception as e:
        print(f"⚠️ ChatGPT 1M error: {e}")
        return 0


# ============================================================
# MAIN
# ============================================================
def process_all_massive():
    print("=" * 60)
    print("🚀 PROCESSING MASSIVE DATASETS + CONVERSATIONS")
    print("=" * 60)

    total_samples = 0

    # Knowledge / general
    total_samples += process_c4()
    total_samples += process_the_pile()
    total_samples += process_openwebmath()
    total_samples += process_wikipedia()

    # Conversations
    total_samples += process_ultrachat()
    total_samples += process_sharegpt()
    total_samples += process_openhermes()
    total_samples += process_chatgpt_multi_turn()

    # Combine
    print("\n" + "=" * 60)
    print("📂 COMBINING ALL CORPORA")
    print("=" * 60)

    files = [
        "c4_corpus.txt",
        "the_pile_corpus.txt",
        "openwebmath_corpus.txt",
        "wikipedia_corpus.txt",
        "ultrachat_conversations.txt",
        "sharegpt_conversations.txt",
        "openhermes_conversations.txt",
        "chatgpt_1m_conversations.txt",
    ]

    with open("massive_corpus.txt", "w", encoding="utf-8") as outfile:
        for file in files:
            if os.path.exists(file) and os.path.getsize(file) > 0:
                with open(file, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\n")
                size = os.path.getsize(file)
                print(f"✅ Added {file} ({size / 1024 / 1024:.2f} MB)")
            else:
                print(f"⚠️ {file} not found or empty")

    print("\n" + "=" * 60)
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📊 Total samples: {total_samples}")

    if os.path.exists("massive_corpus.txt"):
        size = os.path.getsize("massive_corpus.txt")
        print(f"📦 Combined corpus: {size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    if not os.path.exists("massive_corpus.txt"):
        process_all_massive()
    else:
        print("✅ massive_corpus.txt already exists!")
        print(f"📊 File size: {os.path.getsize('massive_corpus.txt') / 1024 / 1024:.2f} MB")