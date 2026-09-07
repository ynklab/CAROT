"""Alternate answer-token SFT and CAROT alignment updates with LoRA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, set_seed

from carot import CAROT, LayerArtifacts, OTConfig, carot_alignment_loss

from .data import load_dataset
from .io import load_yaml
from .modeling import content_token_mask, decoder_layers, load_model_and_tokenizer, logger
from .pairing import build_source_lookup, pair_target_batch


LOG = logger(__name__)


def lm_loss(inputs: dict, logits: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        inputs["labels"][:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def capture_layer(model, layer_index: int, inputs: dict, *, gradients: bool):
    captured = {}
    def hook(_module, _args, output):
        captured["states"] = output[0] if isinstance(output, tuple) else output
    handle = decoder_layers(model)[layer_index].register_forward_hook(hook)
    context = torch.enable_grad() if gradients else torch.no_grad()
    with context:
        model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
    handle.remove()
    return captured["states"]


def alignment_stream(config: dict, tokenizer):
    while True:
        for source in config["alignment_sources"]:
            common = dict(
                mode="embedding",
                batch_size=config["batch_size"],
                split=source["split"],
                max_samples_per_lang=source.get("max_samples_per_language"),
                max_token_length=source.get("max_tokens", 64),
                logger=LOG,
                shuffle=False,
            )
            source_loader = load_dataset(
                source["name"], tokenizer,
                target_langs=[config.get("source_language", "English")], **common,
            )
            target_loader = load_dataset(
                source["name"], tokenizer,
                target_langs=config["target_languages"], **common,
            )
            lookup = build_source_lookup(source_loader)
            for target_batch in target_loader:
                paired_source, paired_target = pair_target_batch(target_batch, lookup, tokenizer)
                if paired_source is not None:
                    yield paired_source, paired_target


def optimizer_step(model, optimizer, scheduler, clip: float | None):
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def run(config: dict) -> None:
    set_seed(config.get("seed", 42))
    model, tokenizer = load_model_and_tokenizer(config["model"])
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora.get("rank", 64),
        lora_alpha=lora.get("alpha", 64),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=lora.get("target_modules", "all-linear"),
        modules_to_save=lora.get("modules_to_save"),
    ))
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.config.use_cache = False

    artifact = LayerArtifacts.load(config["artifact"], map_location=model.device)
    method = CAROT(artifact, OTConfig(**config.get("ot", {}))).to(
        device=model.device, dtype=model.dtype
    )
    layer = config["layer"]
    if artifact.metadata.get("decoder_layer", layer) != layer:
        raise ValueError("Training layer does not match the calibration artifact")

    lm_loader = load_dataset(
        config.get("lm_dataset", "aya_dataset"), tokenizer,
        mode="training", batch_size=config.get("batch_size", 16),
        target_langs=[config.get("source_language", "English")] + config["target_languages"],
        split=config.get("lm_split", "train"),
        max_samples_per_lang=config.get("max_instruction_samples_per_language", 2000),
        max_token_length=config.get("max_instruction_tokens", 500),
        logger=LOG, shuffle=True,
    )
    epochs = config.get("epochs", 1)
    interval = config.get("alignment_interval", 1)
    alignment_fraction = config.get("alignment_fraction", 0.5)
    lm_steps = epochs * len(lm_loader)
    alignment_steps = math.ceil(lm_steps * alignment_fraction / interval)
    optimizer = AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=config.get("weight_decay", 0.01),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.get("warmup_steps", 300),
        num_training_steps=lm_steps + alignment_steps,
    )
    pairs = alignment_stream(config, tokenizer)
    output_dir = Path(config["output_dir"])
    global_lm_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        for batch in lm_loader:
            lm_inputs = {
                key: value.to(model.device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "labels"}
            }
            output = model(
                input_ids=lm_inputs["input_ids"],
                attention_mask=lm_inputs["attention_mask"],
                use_cache=False,
            )
            loss = lm_loss(lm_inputs, output.logits)
            loss.backward()
            optimizer_step(model, optimizer, scheduler, config.get("gradient_clip", 1.0))
            global_lm_step += 1

            alignment_loss = None
            if (
                global_lm_step % interval == 0
                and global_lm_step <= int(lm_steps * alignment_fraction)
            ):
                source_batch, target_batch = next(pairs)
                source_inputs = {
                    key: value.to(model.device)
                    for key, value in source_batch.items()
                    if key in {"input_ids", "attention_mask"}
                }
                target_inputs = {
                    key: value.to(model.device)
                    for key, value in target_batch.items()
                    if key in {"input_ids", "attention_mask"}
                }
                source_states = capture_layer(model, layer, source_inputs, gradients=False)
                target_states = capture_layer(model, layer, target_inputs, gradients=True)
                source_mask = content_token_mask(
                    source_inputs["input_ids"], source_inputs["attention_mask"], tokenizer,
                    source_batch.get("question_mask", None).to(model.device)
                    if source_batch.get("question_mask") is not None else None,
                )
                target_mask = content_token_mask(
                    target_inputs["input_ids"], target_inputs["attention_mask"], tokenizer,
                    target_batch.get("question_mask", None).to(model.device)
                    if target_batch.get("question_mask") is not None else None,
                )
                sample_losses = []
                for index in range(target_states.shape[0]):
                    sample_losses.append(carot_alignment_loss(
                        source_states[index, source_mask[index]],
                        target_states[index, target_mask[index]],
                        method.eraser,
                        method.whitening,
                        method.config,
                    ))
                alignment_loss = torch.stack(sample_losses).mean()  # average over batch
                alignment_loss.backward()
                optimizer_step(model, optimizer, scheduler, config.get("gradient_clip", 1.0))
            if global_lm_step % config.get("logging_steps", 50) == 0:
                LOG.info(
                    "epoch=%d lm_step=%d/%d lm_loss=%.5f alignment_loss=%.5f lr=%.2e",
                    epoch + 1, global_lm_step, lm_steps, loss.item(),
                    alignment_loss.item() if alignment_loss is not None else 0.0, scheduler.get_last_lr()[0],
                )

        checkpoint = output_dir / f"epoch_{epoch + 1}"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        LOG.info("Saved %s", checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(load_yaml(parser.parse_args().config))


if __name__ == "__main__":
    main()
