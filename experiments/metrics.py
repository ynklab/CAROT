from __future__ import annotations

import re

from .data.normalize import normalize_answer
from .data.prompts import ANSWER_KEYS


def extract_answer(dataset: str, generation: str, label, language: str):
    answer_word = ANSWER_KEYS.get(language, "Answer")
    text = str(generation).strip()
    lines = text.splitlines()
    if dataset in {"gmmlu", "belebele"}:
        pattern = rf"(?:{re.escape(answer_word)}|[Aa]nswer)(?:\s*is)?\s*[:：]?\s*[\(\[]?([A-D])[\)\]]?"
        for line in reversed(lines):
            match = re.search(pattern, re.sub(r"[*_`]+", "", line))
            if match:
                return match.group(1).upper()
        for line in reversed(lines):
            match = re.search(r"\b([A-D])\b", re.sub(r"[*_`]+", "", line))
            if match:
                return match.group(1).upper()
    elif dataset == "klar":
        return label if label is not None and re.search(re.escape(str(label)), text, re.I) else None
    elif dataset == "xquad":
        return normalize_answer(text, lang=language)
    elif dataset not in {"gmmlu", "belebele", "klar"}:
        raise ValueError(f"Unsupported evaluation dataset: {dataset}")
    return None


def load_language_identifier():
    try:
        import fasttext
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
    return fasttext.load_model(path)


def language_fidelity(identifier, text: str, expected_code: str):
    if identifier is None:
        return None
    labels, _ = identifier.predict(text.replace("\n", " ") or " ", k=1)
    predicted = labels[0].removeprefix("__label__").split("_")[0]
    return int(predicted == expected_code.split("_")[0])
