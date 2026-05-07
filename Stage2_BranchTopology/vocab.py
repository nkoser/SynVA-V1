from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def build_branch_skeleton_vocab(max_incoming_length: int) -> Dict[str, object]:
    max_incoming_length = int(max_incoming_length)
    if max_incoming_length < 0:
        raise ValueError("max_incoming_length must be >= 0.")

    len_token_ids = {int(length): int(length) for length in range(max_incoming_length + 1)}
    degree_offset = max_incoming_length + 1
    degree_token_ids = {0: degree_offset, 1: degree_offset + 1, 2: degree_offset + 2}
    bos_token_id = degree_offset + 3
    eos_token_id = degree_offset + 4
    pad_token_id = degree_offset + 5

    return {
        "representation": "branch_skeleton_preorder_v1",
        "tokens_per_event": 2,
        "max_incoming_length": int(max_incoming_length),
        "len_token_ids": len_token_ids,
        "degree_token_ids": degree_token_ids,
        "bos_token_id": int(bos_token_id),
        "eos_token_id": int(eos_token_id),
        "pad_token_id": int(pad_token_id),
        "vocab_size": int(pad_token_id + 1),
    }


def encode_branch_skeleton(
    incoming_lengths: Sequence[int],
    degrees: Sequence[int],
    vocab: Dict[str, object],
) -> List[int]:
    lengths = [int(v) for v in incoming_lengths]
    degs = [int(v) for v in degrees]
    if len(lengths) != len(degs):
        raise ValueError("incoming_lengths and degrees must have the same length.")
    body: List[int] = []
    len_token_ids = {int(k): int(v) for k, v in dict(vocab["len_token_ids"]).items()}
    degree_token_ids = {int(k): int(v) for k, v in dict(vocab["degree_token_ids"]).items()}
    for length, degree in zip(lengths, degs):
        if length not in len_token_ids:
            raise KeyError(f"Incoming branch length {length} is missing from the vocabulary.")
        if degree not in degree_token_ids:
            raise KeyError(f"Degree {degree} is missing from the vocabulary.")
        body.append(int(len_token_ids[length]))
        body.append(int(degree_token_ids[degree]))
    return [int(vocab["bos_token_id"])] + body + [int(vocab["eos_token_id"])]


def decode_branch_skeleton(tokens: Sequence[int], vocab: Dict[str, object]) -> Tuple[List[int], List[int]]:
    bos_token_id = int(vocab["bos_token_id"])
    eos_token_id = int(vocab["eos_token_id"])
    pad_token_id = int(vocab["pad_token_id"])
    seq = [int(v) for v in tokens if int(v) != pad_token_id]
    if not seq:
        raise ValueError("Token sequence is empty.")
    if seq[0] == bos_token_id:
        seq = seq[1:]
    if not seq:
        raise ValueError("Token sequence only contained BOS.")
    if eos_token_id in seq:
        eos_index = seq.index(eos_token_id)
        seq = seq[:eos_index]
    if len(seq) % 2 != 0:
        raise ValueError("Branch-skeleton token body length must be divisible by 2.")

    len_from_id = {int(v): int(k) for k, v in dict(vocab["len_token_ids"]).items()}
    degree_from_id = {int(v): int(k) for k, v in dict(vocab["degree_token_ids"]).items()}

    lengths: List[int] = []
    degrees: List[int] = []
    for idx in range(0, len(seq), 2):
        len_token = int(seq[idx])
        degree_token = int(seq[idx + 1])
        if len_token not in len_from_id:
            raise KeyError(f"Token {len_token} is not a valid incoming-length token.")
        if degree_token not in degree_from_id:
            raise KeyError(f"Token {degree_token} is not a valid degree token.")
        lengths.append(int(len_from_id[len_token]))
        degrees.append(int(degree_from_id[degree_token]))
    return lengths, degrees


def allowed_length_token_ids(vocab: Dict[str, object]) -> List[int]:
    return [int(v) for _, v in sorted(dict(vocab["len_token_ids"]).items(), key=lambda item: int(item[0]))]


def allowed_degree_token_ids(vocab: Dict[str, object], allow_degree_one: bool) -> List[int]:
    degree_token_ids = {int(k): int(v) for k, v in dict(vocab["degree_token_ids"]).items()}
    allowed = [degree_token_ids[0], degree_token_ids[2]]
    if allow_degree_one:
        allowed.insert(1, degree_token_ids[1])
    return allowed

