import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
import warnings
from tqdm import tqdm

from transformer import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import BilingualDataset, causal_mask


# ---------------------------------------------------------------------------
# TrainTokenizer — matches the usage in main.py
# ---------------------------------------------------------------------------

class TrainTokenizer:
    """
    Wraps HuggingFace `tokenizers` BPE tokenizer with a simple API:

        trainer = TrainTokenizer()
        files   = trainer.data_loader("data.txt")
        trainer.train_(files)
        trainer.save("tokenizer.json")
        encoded = trainer.encode("Hello world")
        print(encoded.tokens)
        decoded = trainer.decode(encoded.ids)
        print(decoded)
    """

    def __init__(self, vocab_size: int = 10_000, min_frequency: int = 2):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        )
        self._trained = False

    # ------------------------------------------------------------------
    # data_loader: accepts a txt file path (or list of paths) and returns
    # a list of file paths suitable for tokenizer.train()
    # ------------------------------------------------------------------
    def data_loader(self, path) -> list[str]:
        if isinstance(path, (str, Path)):
            paths = [str(path)]
        else:
            paths = [str(p) for p in path]
        return paths

    def train_(self, files: list[str]):
        """Train BPE tokenizer on the given file(s)."""
        self.tokenizer.train(files, self.trainer)
        self._trained = True

    def save(self, path: str):
        if not self._trained:
            raise RuntimeError("Call train_() before save().")
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "TrainTokenizer":
        obj = cls.__new__(cls)
        obj.tokenizer = Tokenizer.from_file(path)
        obj._trained = True
        return obj

    def encode(self, text: str):
        """Returns a tokenizers.Encoding with .tokens and .ids attributes."""
        if not self._trained:
            raise RuntimeError("Call train_() before encode().")
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        if not self._trained:
            raise RuntimeError("Call train_() before decode().")
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)


# ---------------------------------------------------------------------------
# Helper: build (or load) a tokenizer from a HuggingFace dataset split
# ---------------------------------------------------------------------------

def get_or_build_tokenizer(config: dict, ds, lang: str) -> Tokenizer:
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        min_frequency=2,
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
    )

    def get_sentences(ds, lang):
        for item in ds:
            yield item["translation"][lang]

    tokenizer.train_from_iterator(get_sentences(ds, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


# ---------------------------------------------------------------------------
# Build DataLoaders
# ---------------------------------------------------------------------------

def get_ds(config: dict):
    ds_raw = load_dataset(
        config["datasource"],
        f'{config["lang_src"]}-{config["lang_tgt"]}',
        split="train",
    )

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # 90/10 train-val split
    train_size = int(0.9 * len(ds_raw))
    val_size   = len(ds_raw) - train_size
    train_raw, val_raw = random_split(ds_raw, [train_size, val_size])

    train_ds = BilingualDataset(
        train_raw, tokenizer_src, tokenizer_tgt,
        config["lang_src"], config["lang_tgt"], config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_raw, tokenizer_src, tokenizer_tgt,
        config["lang_src"], config["lang_tgt"], config["seq_len"],
    )

    # Warn about sequences that exceed seq_len
    max_src = max(len(tokenizer_src.encode(item["translation"][config["lang_src"]]).ids) for item in ds_raw)
    max_tgt = max(len(tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids) for item in ds_raw)
    print(f"Max src token length: {max_src}")
    print(f"Max tgt token length: {max_tgt}")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,                   shuffle=False)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def get_model(config: dict, src_vocab_size: int, tgt_vocab_size: int):
    return build_transformer(
        src_vocab_size=src_vocab_size,
        target_vocab_size=tgt_vocab_size,
        src_seq_length=config["seq_len"],
        targt_seq_length=config["seq_len"],
        N=config["N"],
        h=config["h"],
        d_model=config["d_model"],
        dropout=config["dropout"],
        d_ff=config["d_ff"],
    )


# ---------------------------------------------------------------------------
# Greedy decode (for validation)
# ---------------------------------------------------------------------------

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)

    # Start with [SOS]
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])          # (1, vocab_size)
        next_token = torch.argmax(prob, dim=-1)   # greedy pick

        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source).fill_(next_token.item()).to(device),
        ], dim=1)

        if next_token == eos_idx:
            break

    return decoder_input.squeeze(0)


# ---------------------------------------------------------------------------
# Validation loop (prints a few examples)
# ---------------------------------------------------------------------------

def run_validation(model, val_loader, tokenizer_tgt, max_len, device, num_examples=2):
    model.eval()
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask  = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Validation batch size must be 1"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)
            src_text   = batch["src_text"][0]
            tgt_text   = batch["tgt_text"][0]
            pred_text  = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            print("-" * 60)
            print(f"SOURCE   : {src_text}")
            print(f"TARGET   : {tgt_text}")
            print(f"PREDICTED: {pred_text}")

            count += 1
            if count >= num_examples:
                break


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_model(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(
        config,
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=config["eps"])
    loss_fn   = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        label_smoothing=0.1,
    )

    initial_epoch = 0
    global_step   = 0

    # ── Optional: resume from checkpoint ────────────────────────────────
    preload = config.get("preload")
    if preload == "latest":
        ckpt_path = latest_weights_file_path(config)
    elif preload is not None:
        ckpt_path = get_weights_file_path(config, preload)
    else:
        ckpt_path = None

    if ckpt_path and Path(ckpt_path).exists():
        print(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step   = state["global_step"]

    # ── Training epochs ──────────────────────────────────────────────────
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input  = batch["encoder_input"].to(device)   # (B, seq_len)
            decoder_input  = batch["decoder_input"].to(device)   # (B, seq_len)
            encoder_mask   = batch["encoder_mask"].to(device)    # (B, 1, 1, seq_len)
            decoder_mask   = batch["decoder_mask"].to(device)    # (B, 1, seq_len, seq_len)
            label          = batch["label"].to(device)           # (B, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output    = model.project(decoder_output)       # (B, seq_len, vocab_size)

            # Reshape for CrossEntropyLoss: (B*seq_len, vocab_size) vs (B*seq_len,)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1),
            )

            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        # Validate after each epoch
        run_validation(model, val_loader, tokenizer_tgt, config["seq_len"], device)

        # Save checkpoint
        ckpt_path = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch":                epoch,
            "global_step":          global_step,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")
