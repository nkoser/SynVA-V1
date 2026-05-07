from __future__ import annotations

import os
from typing import List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, folder_path: str, limit: Optional[int] = None):
        files = [
            os.path.join(folder_path, name)
            for name in sorted(os.listdir(folder_path))
            if name.endswith(".tok") and not name.startswith(".")
        ]
        if limit is not None:
            files = files[: int(limit)]
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = torch.load(self.files[idx], map_location="cpu")
        if not torch.is_tensor(seq):
            seq = torch.as_tensor(seq)
        return seq.long().view(-1)


def collate_tokens(batch: List[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    return pad_sequence(batch, batch_first=True, padding_value=int(pad_token_id))

