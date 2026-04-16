import torch
from torch.utils.data import Dataset


def causal_mask(size: int) -> torch.Tensor:
    """
    Returns a (1, size, size) boolean mask where the upper triangle is False
    (masked out) so each position can only attend to itself and earlier positions.
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0   # True = allowed to attend


class BilingualDataset(Dataset):
    """
    Wraps a HuggingFace dataset split for sequence-to-sequence translation.

    Each item returns a dict with:
        encoder_input  : (seq_len,)           — src tokens padded to seq_len
        decoder_input  : (seq_len,)           — [SOS] + tgt tokens, padded
        label          : (seq_len,)           — tgt tokens + [EOS], padded
        encoder_mask   : (1, 1, seq_len)      — 1 where real token, 0 for PAD
        decoder_mask   : (1, seq_len, seq_len)— causal mask ∩ padding mask
        src_text       : raw source string
        tgt_text       : raw target string
    """

    def __init__(
        self,
        ds,
        tokenizer_src,
        tokenizer_tgt,
        src_lang: str,
        tgt_lang: str,
        seq_len: int,
    ):
        super().__init__()
        self.ds          = ds
        self.tok_src     = tokenizer_src
        self.tok_tgt     = tokenizer_tgt
        self.src_lang    = src_lang
        self.tgt_lang    = tgt_lang
        self.seq_len     = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        src_text     = src_tgt_pair["translation"][self.src_lang]
        tgt_text     = src_tgt_pair["translation"][self.tgt_lang]

        src_ids = self.tok_src.encode(src_text).ids
        tgt_ids = self.tok_tgt.encode(tgt_text).ids

        # How many PAD tokens are needed?
        # encoder_input = [SOS] + src + [EOS] + padding
        enc_num_padding = self.seq_len - len(src_ids) - 2
        # decoder_input  = [SOS] + tgt + padding   (no EOS)
        # label          = tgt  + [EOS] + padding   (no SOS)
        dec_num_padding = self.seq_len - len(tgt_ids) - 1

        if enc_num_padding < 0 or dec_num_padding < 0:
            raise ValueError(
                f"Sequence too long: src={len(src_ids)}, tgt={len(tgt_ids)}, "
                f"seq_len={self.seq_len}"
            )

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(src_ids, dtype=torch.int64),
            self.eos_token,
            torch.full((enc_num_padding,), self.pad_token.item(), dtype=torch.int64),
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_ids, dtype=torch.int64),
            torch.full((dec_num_padding,), self.pad_token.item(), dtype=torch.int64),
        ])

        label = torch.cat([
            torch.tensor(tgt_ids, dtype=torch.int64),
            self.eos_token,
            torch.full((dec_num_padding,), self.pad_token.item(), dtype=torch.int64),
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0)         == self.seq_len

        # Padding masks: (1, 1, seq_len) — True where NOT padding
        encoder_mask = (encoder_input != self.pad_token.item()).unsqueeze(0).unsqueeze(0).int()

        # Decoder: combine padding mask with causal mask — (1, seq_len, seq_len)
        decoder_pad_mask = (decoder_input != self.pad_token.item()).unsqueeze(0).unsqueeze(0).int()
        decoder_mask     = decoder_pad_mask & causal_mask(self.seq_len)

        return {
            "encoder_input": encoder_input,    # (seq_len,)
            "decoder_input": decoder_input,    # (seq_len,)
            "label":         label,            # (seq_len,)
            "encoder_mask":  encoder_mask,     # (1, 1, seq_len)
            "decoder_mask":  decoder_mask,     # (1, seq_len, seq_len)
            "src_text":      src_text,
            "tgt_text":      tgt_text,
        }
