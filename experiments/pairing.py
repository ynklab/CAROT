from __future__ import annotations

import torch


def build_source_lookup(loader) -> dict:
    lookup = {}
    for batch in loader:
        instruction_ids = batch.get("instruction_ids")
        question_masks = batch.get("question_masks")
        for index, raw_id in enumerate(batch["ids"]):
            key = (
                (str(raw_id), str(instruction_ids[index]))
                if instruction_ids is not None
                else str(raw_id)
            )
            row = {
                "input_ids": batch["input_ids"][index].clone(),
                "attention_mask": batch["attention_mask"][index].clone(),
            }
            if question_masks:
                row["question_mask"] = question_masks[0][index].clone()
            lookup[key] = row
    return lookup


def _pad(rows: list[torch.Tensor], value: int, side: str) -> torch.Tensor:
    width = max(row.shape[0] for row in rows)
    padded = []
    for row in rows:
        amount = width - row.shape[0]
        fill = row.new_full((amount,), value)
        padded.append(torch.cat((fill, row)) if side == "left" else torch.cat((row, fill)))
    return torch.stack(padded)


def pair_target_batch(target_batch: dict, source_lookup: dict, tokenizer):
    instruction_ids = target_batch.get("instruction_ids")
    selected = []
    keys = []
    for index, raw_id in enumerate(target_batch["ids"]):
        key = (
            (str(raw_id), str(instruction_ids[index]))
            if instruction_ids is not None
            else str(raw_id)
        )
        if key in source_lookup:
            selected.append(index)
            keys.append(key)
    if not selected:
        return None, None

    source_rows = [source_lookup[key] for key in keys]
    pad_id = tokenizer.pad_token_id or 0
    side = tokenizer.padding_side
    source = {
        "input_ids": _pad([row["input_ids"] for row in source_rows], pad_id, side),
        "attention_mask": _pad([row["attention_mask"] for row in source_rows], 0, side),
    }
    if all("question_mask" in row for row in source_rows):
        source["question_mask"] = _pad(
            [row["question_mask"] for row in source_rows], 0, side
        )
    target = {
        "input_ids": target_batch["input_ids"][selected],
        "attention_mask": target_batch["attention_mask"][selected],
    }
    if target_batch.get("question_masks"):
        target["question_mask"] = target_batch["question_masks"][0][selected]
    return source, target

