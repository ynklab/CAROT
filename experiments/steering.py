"""Run CAROT inference-time steering and score paper benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import set_seed

from carot import CAROT, LayerArtifacts, OTConfig

from .data import load_dataset
from .io import load_yaml, write_json, write_jsonl
from .metrics import extract_answer, language_fidelity, load_language_identifier
from .modeling import (
    content_token_mask,
    decoder_layers,
    load_model_and_tokenizer,
    logger,
    move_batch,
)


LOG = logger(__name__)


def _question_mask(batch):
    masks = batch.get("question_masks")
    if not masks:
        return None
    return torch.stack([mask.bool() for mask in masks]).any(dim=0)


def _generated_text(model, tokenizer, inputs, max_new_tokens: int):
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        use_cache=True,
    )
    return tokenizer.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def run_language(model, tokenizer, method: CAROT, config: dict, target_language: str):
    dataset = config["dataset"]
    common = dict(
        mode="evaluation",
        batch_size=config.get("batch_size", 16),
        split=config["split"],
        category=config.get("category"),
        max_samples_per_lang=config.get("max_samples_per_language"),
        logger=LOG,
    )
    source_language = config.get("source_language", "English")
    source_loader = load_dataset(dataset, tokenizer, target_langs=[source_language], **common)
    target_loader = load_dataset(dataset, tokenizer, target_langs=[target_language], **common)
    layer = decoder_layers(model)[config["layer"]]
    identifier = load_language_identifier() if config.get("language_fidelity", True) else None
    rows = []

    for source_batch, target_batch in zip(source_loader, target_loader):
        if [str(x) for x in source_batch["ids"]] != [str(x) for x in target_batch["ids"]]:
            raise ValueError("Source and target batches are not parallel by example ID")
        source_inputs = move_batch(source_batch, model.device)
        target_inputs = move_batch(target_batch, model.device)

        with torch.no_grad():
            source_generations = _generated_text(
                model, tokenizer, source_inputs, config.get("max_new_tokens", 500)
            )

        source_cache = {}
        def cache_source(_module, _inputs, output):
            source_cache["states"] = (output[0] if isinstance(output, tuple) else output).detach()
        handle = layer.register_forward_hook(cache_source)
        with torch.no_grad():
            model(**source_inputs, use_cache=False)
        handle.remove()

        source_question = _question_mask(source_batch)
        target_question = _question_mask(target_batch)
        source_mask = content_token_mask(
            source_inputs["input_ids"], source_inputs["attention_mask"], tokenizer,
            source_question.to(model.device) if source_question is not None else None,
        )
        target_mask = content_token_mask(
            target_inputs["input_ids"], target_inputs["attention_mask"], tokenizer,
            target_question.to(model.device) if target_question is not None else None,
        )

        applied = {"value": False}
        def steer_target(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not applied["value"]:
                # Steering is applied only in the prefill phase, not during generation.
                steered = hidden.clone()
                for index in range(hidden.shape[0]):
                    steered[index, target_mask[index]] = method(
                        source_cache["states"][index, source_mask[index]],
                        hidden[index, target_mask[index]],
                    )
                hidden = steered
                applied["value"] = True
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        handle = layer.register_forward_hook(steer_target)
        with torch.no_grad():
            target_generations = _generated_text(
                model, tokenizer, target_inputs, config.get("max_new_tokens", 500)
            )
        handle.remove()

        for index, (source_text, target_text) in enumerate(
            zip(source_generations, target_generations)
        ):
            label = target_batch.get("labels", [None] * len(target_generations))[index]
            target_prediction = extract_answer(dataset, target_text, label, target_language)
            source_prediction = extract_answer(dataset, source_text, label, source_language)
            expected_code = target_batch["src_langs"][index]
            rows.append({
                "id": target_batch["ids"][index],
                "language": target_language,
                "label": label,
                "prediction": target_prediction,
                "source_prediction": source_prediction,
                "is_correct": int(str(target_prediction).strip() == str(label).strip()),
                "agreement": int(str(target_prediction).strip() == str(source_prediction).strip()),
                "fidelity": language_fidelity(identifier, target_text, expected_code),
                "generation": target_text,
                "source_generation": source_text,
            })
    return rows


def run(config: dict) -> None:
    set_seed(config.get("seed", 42))
    model, tokenizer = load_model_and_tokenizer(
        config["model"], adapter=config.get("adapter")
    )
    artifacts = LayerArtifacts.load(config["artifact"], map_location=model.device)
    artifact_layer = artifacts.metadata.get("decoder_layer")
    if artifact_layer is not None and artifact_layer != config["layer"]:
        raise ValueError(f"Artifact is for layer {artifact_layer}, not {config['layer']}")
    method = CAROT(artifacts, OTConfig(**config.get("ot", {}))).to(
        device=model.device, dtype=model.dtype
    )
    output_dir = Path(config["output_dir"])
    summaries = {}
    for language in config["target_languages"]:
        LOG.info("Steering %s -> %s", config.get("source_language", "English"), language)
        rows = run_language(model, tokenizer, method, config, language)
        write_jsonl(output_dir / f"{language}.jsonl", rows)
        valid_fidelity = [row["fidelity"] for row in rows if row["fidelity"] is not None]
        summaries[language] = {
            "samples": len(rows),
            "accuracy": sum(row["is_correct"] for row in rows) / max(len(rows), 1),
            "agreement": sum(row["agreement"] for row in rows) / max(len(rows), 1),
            "fidelity": sum(valid_fidelity) / len(valid_fidelity) if valid_fidelity else None,
        }
        LOG.info(
            "Summary for %s: %d samples, accuracy %.3f, agreement %.3f, fidelity %.3f",
            language,
            summaries[language]["samples"],
            summaries[language]["accuracy"],
            summaries[language]["agreement"],
            summaries[language]["fidelity"],
        )
    write_json(output_dir / "summary.json", summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(load_yaml(parser.parse_args().config))


if __name__ == "__main__":
    main()
