from __future__ import annotations

from typing import Dict, List

import torch
from transformers import LogitsProcessor

from Stage2_BranchTopology.vocab import allowed_degree_token_ids, allowed_length_token_ids


class BranchSkeletonConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab: Dict[str, object]):
        self.vocab = vocab
        self.bos_token_id = int(vocab["bos_token_id"])
        self.eos_token_id = int(vocab["eos_token_id"])
        self.pad_token_id = int(vocab["pad_token_id"])
        len_token_ids = {int(k): int(v) for k, v in dict(vocab["len_token_ids"]).items()}
        self.len_zero_token_id = int(len_token_ids[0])
        self.length_token_ids = [int(v) for v in allowed_length_token_ids(vocab)]
        self.root_degree_token_ids = [int(v) for v in allowed_degree_token_ids(vocab, allow_degree_one=True)]
        self.non_root_degree_token_ids = [int(v) for v in allowed_degree_token_ids(vocab, allow_degree_one=False)]

    def _body_tokens(self, seq: torch.Tensor) -> List[int]:
        items = []
        for token in seq.tolist():
            token = int(token)
            if token == self.pad_token_id:
                continue
            if token == self.bos_token_id:
                continue
            if token == self.eos_token_id:
                break
            items.append(token)
        return items

    def _pending_events(self, body_tokens: List[int]) -> int:
        degree_from_id = {int(v): int(k) for k, v in dict(self.vocab["degree_token_ids"]).items()}
        pending = 1
        for idx in range(1, len(body_tokens), 2):
            token = int(body_tokens[idx])
            if token not in degree_from_id:
                continue
            degree = int(degree_from_id[token])
            pending = pending - 1 + degree
        return pending

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.finfo(scores.dtype).min
        for row in range(scores.shape[0]):
            body = self._body_tokens(input_ids[row])
            pending = self._pending_events(body)
            slot_idx = len(body) % 2
            event_count = len(body) // 2

            allowed: List[int]
            if slot_idx == 0:
                if pending <= 0:
                    allowed = [self.eos_token_id]
                elif event_count == 0:
                    allowed = [self.len_zero_token_id]
                else:
                    allowed = list(self.length_token_ids)
            else:
                if event_count == 0:
                    allowed = list(self.root_degree_token_ids)
                else:
                    allowed = list(self.non_root_degree_token_ids)

            mask = torch.full_like(scores[row], neg_inf)
            mask[allowed] = scores[row, allowed]
            scores[row] = mask
        return scores
