"""Extract calibration representations and fit CAROT layer artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from transformers import set_seed

from carot import fit_layer_artifacts, pool_sentence_embeddings

from .data import load_dataset
from .data.language_codes import LANG_CODE_MAPS
from .io import load_yaml, write_json
from .modeling import content_token_mask, load_model_and_tokenizer, logger


LOG = logger(__name__)


def extract_source(model, tokenizer, source: dict, languages: list[str], layers: list[int]):
    loader = load_dataset(
        source["name"],
        tokenizer,
        mode="embedding",
        batch_size=source.get("batch_size", 16),
        target_langs=languages,
        split=source["split"],
        max_samples_per_lang=source.get("max_samples_per_language"),
        max_token_length=source.get("max_tokens", 64),
        logger=LOG,
    )
    reverse_codes = {code: language for language, code in LANG_CODE_MAPS[source["name"]].items()}
    collected: dict[int, dict[str, list[torch.Tensor]]] = {
        layer: defaultdict(list) for layer in layers
    }
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            selected = content_token_mask(input_ids, attention_mask, tokenizer)
            for layer in layers:
                # Transformers hidden_states[0] is the embedding output.
                sentence = pool_sentence_embeddings(
                    outputs.hidden_states[layer + 1], selected, method="position_weighted"
                ).float().cpu()
                for row, code in zip(sentence, batch["languages"]):
                    language = reverse_codes.get(code)
                    if language in languages:
                        collected[layer][language].append(row)
            if batch_index % 50 == 0:
                LOG.info("%s/%s: extracted %d batches", source["name"], source["split"], batch_index)
    return {
        layer: {language: torch.stack(rows) for language, rows in by_language.items()}
        for layer, by_language in collected.items()
    }


def run(config: dict) -> None:
    set_seed(config.get("seed", 42))
    model_name = config["model"]
    languages = config["languages"]
    layers = config["layers"]
    output_dir = Path(config["output_dir"])
    embeddings_dir = output_dir / "embeddings"
    artifacts_dir = output_dir / "artifacts"
    model, tokenizer = load_model_and_tokenizer(model_name, use_chat_template=False)

    all_embeddings: dict[int, dict[str, list[torch.Tensor]]] = {
        layer: defaultdict(list) for layer in layers
    }
    manifest_sources = []
    for source in config["sources"]:
        LOG.info("Extracting %s/%s", source["name"], source["split"])
        extracted = extract_source(model, tokenizer, source, languages, layers)
        source_dir = embeddings_dir / source["name"] / source["split"]
        for layer, by_language in extracted.items():
            for language, values in by_language.items():
                all_embeddings[layer][language].append(values)
                path = source_dir / f"layer_{layer}" / f"{language}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(values, path)
        manifest_sources.append({k: v for k, v in source.items() if k != "batch_size"})

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for layer in layers:
        missing = [language for language in languages if not all_embeddings[layer][language]]
        if missing:
            raise ValueError(f"Layer {layer}: no calibration embeddings for {missing}")
        parts = []
        labels = []
        for language_index, language in enumerate(languages):
            values = torch.cat(all_embeddings[layer][language], dim=0)
            parts.append(values)
            labels.append(torch.full((values.shape[0],), language_index, dtype=torch.long))
        embeddings = torch.cat(parts).to(config.get("fit_device", "cpu"))
        language_labels = torch.cat(labels).to(embeddings.device)
        metadata = {
            "model": model_name,
            "model_revision": config.get("model_revision"),
            "tokenizer": model_name,
            "decoder_layer": layer,
            "hidden_state_index": layer + 1,
            "pooling": "position_weighted",
            "excluded_tokens": "padding,bos,eos,attention_sink_if_no_bos",
            "languages": languages,
            "sources": manifest_sources,
            "samples_per_language": {
                language: sum(part.shape[0] for part in all_embeddings[layer][language])
                for language in languages
            },
        }
        artifact = fit_layer_artifacts(
            embeddings,
            language_labels,
            eigenvalue_relative_threshold=config.get("eigenvalue_relative_threshold", 1e-5),
            metadata=metadata,
        )
        artifact_path = artifacts_dir / f"layer_{layer}.pt"
        artifact.save(artifact_path)
        LOG.info("Saved layer %d artifact to %s", layer, artifact_path)

    write_json(output_dir / "manifest.json", {
        "model": model_name,
        "languages": languages,
        "layers": layers,
        "sources": manifest_sources,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
