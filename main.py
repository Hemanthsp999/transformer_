"""
main.py — Entry point

Demonstrates the TrainTokenizer API exactly as written in the original main.py,
then kicks off the full Transformer training pipeline.

Usage:
    pip install torch tokenizers datasets tqdm
    python main.py
"""

from train import TrainTokenizer, train_model
from config import get_config
import os


# ---------------------------------------------------------------------------
# 1. Tokenizer demo  (matches the snippet in the original main.py exactly)
# ---------------------------------------------------------------------------

def tokenizer_demo():
    print("=" * 60)
    print("TOKENIZER DEMO")
    print("=" * 60)

    # Create a tiny demo corpus if data.txt doesn't exist
    if not os.path.exists("data.txt"):
        with open("data.txt", "w") as f:
            f.write(
                "Hello world this is a transformer model built from scratch.\n"
                "Attention is all you need paper implementation.\n"
                "Natural language processing with deep learning.\n"
            )
        print("Created demo data.txt")

    trainer = TrainTokenizer()
    files   = trainer.data_loader("data.txt")
    trainer.train_(files)
    trainer.save("tokenizer.json")

    encoded = trainer.encode("Hello world")
    print("Tokens :", encoded.tokens)

    decoded = trainer.decode(encoded.ids)
    print("Decoded:", decoded)
    print()


# ---------------------------------------------------------------------------
# 2. Full Transformer training
# ---------------------------------------------------------------------------

def main():
    tokenizer_demo()

    print("=" * 60)
    print("TRANSFORMER TRAINING")
    print("=" * 60)

    config = get_config()

    # Optionally tweak for a quick smoke-test:
    # config["num_epochs"] = 1
    # config["batch_size"] = 2

    train_model(config)


if __name__ == "__main__":
    main()
