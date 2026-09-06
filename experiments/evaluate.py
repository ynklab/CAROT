"""Evaluate a base or CAROT-trained model on the paper benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import set_seed

from .data import load_dataset
from .io import load_yaml, write_json, write_jsonl
from .metrics import extract_answer, language_fidelity, load_language_identifier
from .modeling import load_model_and_tokenizer, logger, move_batch


LOG = logger(__name__)


def generate(model, tokenizer, batch, max_new_tokens: int):
    inputs = move_batch(batch, model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            use_cache=True,
        )
    return tokenizer.batch_decode(
        output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def run(config: dict) -> None:
    set_seed(config.get("seed", 42))
    model, tokenizer = load_model_and_tokenizer(
        config["model"], adapter=config.get("adapter")
    )
    model.eval()
    identifier = load_language_identifier() if config.get("language_fidelity", True) else None
    output_dir = Path(config["output_dir"])
    summary = {}
    source_language = config.get("source_language", "English")

    for language in config["target_languages"]:
        common = dict(
            mode="evaluation", batch_size=config.get("batch_size", 16),
            split=config["split"], category=config.get("category"),
            max_samples_per_lang=config.get("max_samples_per_language"), logger=LOG,
        )
        source_loader = load_dataset(
            config["dataset"], tokenizer, target_langs=[source_language], **common
        )
        target_loader = load_dataset(
            config["dataset"], tokenizer, target_langs=[language], **common
        )
        rows = []
        for source_batch, target_batch in zip(source_loader, target_loader):
            if [str(x) for x in source_batch["ids"]] != [str(x) for x in target_batch["ids"]]:
                raise ValueError("Source and target batches are not parallel by example ID")
            source_texts = generate(
                model, tokenizer, source_batch, config.get("max_new_tokens", 500)
            )
            target_texts = generate(
                model, tokenizer, target_batch, config.get("max_new_tokens", 500)
            )
            for index, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
                label = target_batch.get("labels", [None] * len(target_texts))[index]
                prediction = extract_answer(config["dataset"], target_text, label, language)
                source_prediction = extract_answer(
                    config["dataset"], source_text, label, source_language
                )
                rows.append({
                    "id": target_batch["ids"][index],
                    "language": language,
                    "label": label,
                    "prediction": prediction,
                    "source_prediction": source_prediction,
                    "is_correct": int(str(prediction).strip() == str(label).strip()),
                    "agreement": int(str(prediction).strip() == str(source_prediction).strip()),
                    "fidelity": language_fidelity(
                        identifier, target_text, target_batch["src_langs"][index]
                    ),
                    "generation": target_text,
                    "source_generation": source_text,
                })
        write_jsonl(output_dir / f"{language}.jsonl", rows)
        fidelities = [row["fidelity"] for row in rows if row["fidelity"] is not None]
        summary[language] = {
            "samples": len(rows),
            "accuracy": sum(row["is_correct"] for row in rows) / max(len(rows), 1),
            "agreement": sum(row["agreement"] for row in rows) / max(len(rows), 1),
            "fidelity": sum(fidelities) / len(fidelities) if fidelities else None,
        }
        LOG.info("%s: %s", language, summary[language])
    write_json(output_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(load_yaml(parser.parse_args().config))


if __name__ == "__main__":
    main()

