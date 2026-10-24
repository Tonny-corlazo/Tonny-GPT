from pathlib import Path

DATASET_DIR = Path("datasets")
OUTPUT_FILE = Path("training_corpus.txt")

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for category in sorted(DATASET_DIR.iterdir()):

        if category.is_dir():

            for file in sorted(category.glob("*.txt")):

                print(f"Adding {file.name}")

                outfile.write(f"<BEGIN_BOOK>\n")
                outfile.write(f"<CATEGORY={category.name.upper()}>\n")
                outfile.write(f"<BOOK={file.stem.upper()}>\n\n")

                text = file.read_text(encoding="utf-8")

                outfile.write(text)

                outfile.write("\n\n<END>\n\n")