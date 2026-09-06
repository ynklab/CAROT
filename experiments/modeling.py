from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# Add mappings from base models to instruction-tuned variants here.
# Used to load chat templates and EOS tokens from instruction-tuned models into base models
# in the training experiment.
INSTRUCTION_MODEL = {
    "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-4b-pt": "google/gemma-3-4b-it",
    "Qwen/Qwen3-4B-Base": "Qwen/Qwen3-4B",
}


def logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger(name)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_name: str,
    *,
    adapter: str | Path | None = None,
    device: torch.device | None = None,
    use_chat_template: bool = True,
):
    device = device or default_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    template_model = INSTRUCTION_MODEL.get(model_name, model_name)
    template_generation = None
    if use_chat_template and not tokenizer.chat_template:
        template_tokenizer = AutoTokenizer.from_pretrained(template_model, padding_side="left")
        tokenizer.chat_template = template_tokenizer.chat_template
        tokenizer.eos_token_id = template_tokenizer.eos_token_id
        if template_tokenizer.pad_token_id is not None:
            tokenizer.pad_token_id = template_tokenizer.pad_token_id
        try:
            template_generation = GenerationConfig.from_pretrained(template_model)
        except OSError:
            pass

    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    model_kwargs = {"torch_dtype": dtype}
    if "gemma-3" in model_name.lower():
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.to(device)
    if template_generation is not None:
        model.generation_config.eos_token_id = template_generation.eos_token_id
    return model, tokenizer


def decoder_layers(model):
    """Return decoder blocks for the three model families used in the paper."""
    candidates = [
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("base_model", "model", "model", "layers"),
    ]
    for path in candidates:
        value = model
        try:
            for name in path:
                value = getattr(value, name)
            return value
        except AttributeError:
            continue
    if hasattr(model, "get_base_model"):
        return decoder_layers(model.get_base_model())
    raise TypeError(f"Unsupported decoder layout: {type(model).__name__}")


def content_token_mask(input_ids, attention_mask, tokenizer, question_mask=None):
    """Exclude padding, BOS/EOS and the attention-sink token from CAROT."""
    selected = attention_mask.bool()
    if question_mask is not None:
        selected &= question_mask.to(input_ids.device).bool()
    for token_id in {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}:
        if token_id is not None:
            selected &= input_ids.ne(token_id)
    if tokenizer.bos_token_id is None:
        # If there is no BOS token, we assume the first token is a special attention-sink token and exclude it.
        first = attention_mask.long().argmax(dim=1)
        selected[torch.arange(input_ids.shape[0], device=input_ids.device), first] = False
    return selected


def move_batch(batch: dict, device: torch.device, *, labels: bool = False) -> dict:
    names = {"input_ids", "attention_mask"}
    if labels:
        names.add("labels")
    return {name: value.to(device) for name, value in batch.items() if name in names}
