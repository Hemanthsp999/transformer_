from pathlib import Path


def get_config():
    return {
        # Model hyperparameters
        "d_model": 512,
        "N": 6,           # Number of encoder/decoder blocks
        "h": 8,           # Number of attention heads
        "dropout": 0.1,
        "d_ff": 2048,
        "seq_len": 350,   # Max sequence length

        # Training hyperparameters
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,
        "eps": 1e-9,

        # Data / tokenizer
        "lang_src": "en",
        "lang_tgt": "it",
        "tokenizer_file": "tokenizer_{0}.json",
        "datasource": "opus_books",   # HuggingFace dataset

        # Checkpointing
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,              # e.g. "latest" or an epoch number string
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config: dict, epoch: str) -> str:
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config: dict):
    model_folder = Path(config["model_folder"])
    model_files = list(model_folder.glob(f"{config['model_basename']}*.pt"))
    if not model_files:
        return None
    model_files.sort()
    return str(model_files[-1])
