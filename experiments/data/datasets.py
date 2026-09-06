from collections import defaultdict
import json
from pathlib import Path
import random

from datasets import load_dataset as hf_load_dataset, Dataset, concatenate_datasets
from .normalize import normalize_answer
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .language_codes import (
    AYA_LANG_CODE_MAP,
    BELEBELE_LANG_CODE_MAP,
    FLORES_LANG_CODE_MAP,
    GMMLU_LANG_CODE_MAP,
    KLAR_LANG_CODE_MAP,
    NTREX_LANG_CODE_MAP,
    WMTPP_LANG_CODE_MAP,
    XQUAD_LANG_CODE_MAP,
)
from .prompts import BELEBELE_PROMPTS, GMMLU_PROMPTS, MMLU_QUESTION_PART, XQUAD_PROMPTS


def _make_question_mask(input_text, original_text, offsets):
    """Create a binary mask (list of ints) where 1 indicates tokens belonging to original_text.

    Args:
        input_text: Full input string (after apply_chat_template).
        original_text: The problem/question text that should be marked as 1.
        offsets: List of (char_start, char_end) tuples from tokenizer offset_mapping.
    Returns:
        List of 0/1 integers with length == len(offsets).
    """
    char_start = input_text.find(original_text)
    assert char_start != -1, f"Original text not found in input text. Input: '{input_text}', Original: '{original_text}'"
    char_end = char_start + len(original_text)
    return [1 if (tok_end > char_start and tok_start < char_end) else 0
            for tok_start, tok_end in offsets]


def _batch_question_masks(tokenizer, input_texts, original_texts, padded_length):
    """Compute question_mask tensor for a batch, padded to padded_length.

    Args:
        tokenizer: Tokenizer with offset_mapping support.
        input_texts: List of full input strings.
        original_texts: List of original question/problem strings.
        padded_length: Total sequence length after padding (batch dimension 1).
    Returns:
        torch.LongTensor of shape (batch_size, padded_length).
    """
    masks = []
    for text, orig in zip(input_texts, original_texts):
        single = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        mask = _make_question_mask(text, orig.strip(), single['offset_mapping'])
        # Truncate if the per-example tokenization is longer than the (possibly truncated) batch length
        if len(mask) > padded_length:
            mask = mask[:padded_length]
        pad_len = padded_length - len(mask)
        if tokenizer.padding_side == 'left':
            mask = [0] * pad_len + mask
        else:
            mask = mask + [0] * pad_len
        masks.append(mask)
    return torch.tensor(masks, dtype=torch.long)


def _make_sampler(dataset, shuffle=False):
    """Return DistributedSampler if distributed training is active, else None."""
    if torch.distributed.is_initialized():
        return DistributedSampler(dataset, shuffle=shuffle)
    return None


def _batch_labels(tokenizer, input_texts, instruction_texts, input_ids):
    """Build labels tensor for instruction tuning.

    Labels are -100 for instruction tokens (and padding), actual IDs for answer tokens.
    Uses per-example tokenization with offset_mapping to locate the instruction boundary.

    Args:
        tokenizer: Tokenizer (add_special_tokens=False assumed).
        input_texts: List of full sequences (instruction + answer).
        instruction_texts: List of instruction-only strings (without answer).
        input_ids: (B, L) tensor from batched tokenization.
    Returns:
        (B, L) labels tensor.
    """
    labels = input_ids.clone()
    pad_side = tokenizer.padding_side
    L = input_ids.shape[1]

    for i, (full_text, instr_text) in enumerate(zip(input_texts, instruction_texts)):
        single = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = single['offset_mapping']
        n_tokens = len(offsets)
        char_split = len(instr_text)  # char position where answer begins

        # When max_token_length is set, the batched tokenization truncates from the right.
        # Limit processing to the tokens that actually appear in input_ids.
        effective_tokens = min(n_tokens, L)
        pad_len = L - effective_tokens

        if pad_side == 'left':
            labels[i, :pad_len] = -100  # padding on the left
            for j in range(effective_tokens):
                tok_start, _ = offsets[j]
                if tok_start < char_split:
                    labels[i, pad_len + j] = -100
        else:
            labels[i, effective_tokens:] = -100  # padding on the right
            for j in range(effective_tokens):
                tok_start, _ = offsets[j]
                if tok_start < char_split:
                    labels[i, j] = -100

    return labels


def load_flores_plus(tokenizer, target_langs=None, split='dev',
                     batch_size=16, logger=None,
                     max_samples_per_lang=None, max_token_length=None, shuffle=False):
    """Load FLORES+ for the representation analysis.

    Args:
        tokenizer: The tokenizer corresponding to the model.
        target_langs: List of language names to include. None means all languages.
        split: Dataset split ('dev', 'devtest', etc.).
        batch_size: Batch size for the DataLoader.
        logger: Optional logger.
        max_samples_per_lang: Filter to samples with id < max_samples_per_lang.
        max_token_length: Maximum token length for truncation.
        shuffle: Whether to shuffle the DataLoader.

    Returns:
        DataLoader with input_ids, attention_mask, ids, and language codes.
    """
    if target_langs is None:
        langs = list(FLORES_LANG_CODE_MAP.keys())
    else:
        if isinstance(target_langs, str):
            target_langs = [target_langs]
        for lang in target_langs:
            if lang not in FLORES_LANG_CODE_MAP:
                raise ValueError(f"Language {lang} not supported in FLORES+.")
        langs = list(set(target_langs))

    datasets_list = []
    for lang in langs:
        lang_code = FLORES_LANG_CODE_MAP[lang]
        ds = hf_load_dataset('openlanguagedata/flores_plus', lang_code, split=split)
        ds = ds.map(lambda x: {'language': lang})
        datasets_list.append(ds)
    full_dataset = concatenate_datasets(datasets_list)

    if max_samples_per_lang is not None:
        full_dataset = full_dataset.filter(lambda x: int(x['id']) < max_samples_per_lang)

    item_count = defaultdict(int)
    for example in full_dataset:
        item_count[example['language']] += 1
    if logger:
        logger.info(f"Loaded FLORES+ with item counts per language: {dict(item_count)}")

    def preprocess(examples):
        texts = examples['text']
        return {
            'text': texts,
            'id': examples['id'],
            'language': [
                FLORES_LANG_CODE_MAP[language] for language in examples['language']
            ],
        }

    processed = full_dataset.map(preprocess, batched=True).sort('id')

    def collate_fn(batch):
        tokenized = tokenizer(
            [example['text'] for example in batch],
            padding='longest',
            truncation=max_token_length is not None,
            max_length=max_token_length,
            return_tensors='pt',
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'ids': [example['id'] for example in batch],
            'languages': [example['language'] for example in batch],
        }

    sampler = _make_sampler(processed, shuffle=shuffle)
    return DataLoader(processed, batch_size=batch_size, sampler=sampler,
                      collate_fn=collate_fn, shuffle=(sampler is None and shuffle))


def load_ntrex(tokenizer, target_langs=None, split='test', mode='embedding',
               batch_size=16, logger=None,
               max_samples_per_lang=None, max_token_length=None, shuffle=False):
    """Load the NTREX-128 dataset for sentence embedding extraction (embedding mode only).

    Args:
        tokenizer: The tokenizer corresponding to the model.
        target_langs: List of language names to include. None means all languages.
        split: Not used (NTREX has only one split); kept for API consistency.
        mode: Only 'embedding' is supported.
        batch_size: Batch size for the DataLoader.
        logger: Optional logger.
        max_samples_per_lang: Maximum number of samples to load per language.
        max_token_length: Maximum token length for truncation.
        shuffle: Whether to shuffle the DataLoader.

    Returns:
        DataLoader with collate keys: input_ids, attention_mask, ids, languages
    """
    if mode != 'embedding':
        raise ValueError(f"load_ntrex only supports mode='embedding', got '{mode}'")

    if target_langs is None:
        langs = list(NTREX_LANG_CODE_MAP.keys())
    else:
        if isinstance(target_langs, str):
            target_langs = [target_langs]
        for lang in target_langs:
            if lang not in NTREX_LANG_CODE_MAP:
                raise ValueError(f"Language {lang} not supported in NTREX.")
        langs = list(set(target_langs))

    dataset_dict = {'text': [], 'id': [], 'language': []}
    item_count = defaultdict(int)
    for lang in langs:
        lang_code = NTREX_LANG_CODE_MAP[lang]
        data_file = (Path(__file__).parent.parent.parent / 'NTREX' / 'NTREX-128'
                     / f'newstest2019-ref.{lang_code}.txt')
        if not data_file.exists():
            raise FileNotFoundError(f"NTREX data file {data_file} not found.")
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for item_id, line in enumerate(lines):
            text = line.strip()
            dataset_dict['text'].append(text)
            dataset_dict['id'].append(item_id)
            dataset_dict['language'].append(lang_code)
            item_count[lang_code] += 1
            if max_samples_per_lang is not None and item_id + 1 >= max_samples_per_lang:
                break
    if logger:
        logger.info(f"Loaded NTREX with item counts per language: {dict(item_count)}")
    full_dataset = Dataset.from_dict(dataset_dict)
    full_dataset = full_dataset.sort('id')

    def collate_fn(batch):
        tokenized = tokenizer([ex['text'] for ex in batch], padding='longest',
                               truncation=max_token_length is not None,
                               max_length=max_token_length, return_tensors='pt')
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'ids': [ex['id'] for ex in batch],
            'languages': [ex['language'] for ex in batch],
        }

    sampler = _make_sampler(full_dataset, shuffle=shuffle)
    return DataLoader(full_dataset, batch_size=batch_size, sampler=sampler,
                      collate_fn=collate_fn, shuffle=(sampler is None and shuffle))


def load_wmtpp(tokenizer, target_langs=None, split='train', mode='embedding',
               batch_size=16, logger=None,
               max_samples_per_lang=None, max_token_length=None, shuffle=False):
    """Load the WMT24++ dataset for sentence embedding extraction (embedding mode only).

    Args:
        tokenizer: The tokenizer corresponding to the model.
        target_langs: List of language names to include. None means all languages.
        split: Not used (WMT++ has only 'train'); kept for API consistency.
        mode: Only 'embedding' is supported.
        batch_size: Batch size for the DataLoader.
        logger: Optional logger.
        max_token_length: Maximum token length for truncation.
        max_samples_per_lang: Maximum number of samples to load per language.
        shuffle: Whether to shuffle the DataLoader.

    Returns:
        DataLoader with collate keys: input_ids, attention_mask, ids, languages
    """
    if mode != 'embedding':
        raise ValueError(f"load_wmtpp only supports mode='embedding', got '{mode}'")
    if target_langs is None:
        langs = list(WMTPP_LANG_CODE_MAP.keys())
    else:
        if isinstance(target_langs, str):
            target_langs = [target_langs]
        for lang in target_langs:
            if lang not in WMTPP_LANG_CODE_MAP:
                raise ValueError(f"Language {lang} not supported in WMT++.")
        langs = list(set(target_langs))

    dataset_dict = {'text': [], 'id': [], 'language': []}
    item_count = defaultdict(int)

    # Build a global segment_id → integer index from the English/Arabic pair (canonical order)
    id_map = {}
    cur_id = 0
    ds_ref = hf_load_dataset('google/wmt24pp', 'en-ar_EG', split='train')
    for item in ds_ref:
        if item['is_bad_source']:
            continue
        id_map[item['segment_id']] = cur_id
        cur_id += 1

    for lang in langs:
        lang_code = WMTPP_LANG_CODE_MAP[lang]
        if lang_code == 'en':
            ds = hf_load_dataset('google/wmt24pp', 'en-ar_EG', split='train')
            tgt_column = 'source'
        else:
            ds = hf_load_dataset('google/wmt24pp', lang_code, split='train')
            tgt_column = 'target'
        for item in ds:
            if item['is_bad_source']:
                continue
            if item['segment_id'] not in id_map:
                continue
            text = item[tgt_column]
            dataset_dict['text'].append(text)
            dataset_dict['id'].append(id_map[item['segment_id']])
            dataset_dict['language'].append(lang_code)
            item_count[lang_code] += 1
            if max_samples_per_lang is not None and item_count[lang_code] >= max_samples_per_lang:
                break
    if logger:
        logger.info(f"Loaded WMT++ with item counts per language: {dict(item_count)}")
    full_dataset = Dataset.from_dict(dataset_dict)
    full_dataset = full_dataset.sort('id')

    def collate_fn(batch):
        tokenized = tokenizer([ex['text'] for ex in batch], padding='longest',
                               truncation=max_token_length is not None,
                               max_length=max_token_length, return_tensors='pt')
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'ids': [ex['id'] for ex in batch],
            'languages': [ex['language'] for ex in batch],
        }

    sampler = _make_sampler(full_dataset, shuffle=shuffle)
    return DataLoader(full_dataset, batch_size=batch_size, sampler=sampler,
                      collate_fn=collate_fn, shuffle=(sampler is None and shuffle))


def load_multiling_dataset(dataset_name, tokenizer, mode='evaluation', batch_size=32,
                           target_langs=None, split='test',
                           category=None, max_samples_per_lang=None,
                           logger=None, shuffle=False):
    """Load a multilingual benchmark used in the paper's evaluation.

    Args:
        dataset_name: 'gmmlu', 'klar', 'belebele', or 'xquad'.
        tokenizer: Tokenizer for the model.
        mode: Only 'evaluation' is supported.
        batch_size: Batch size for the DataLoader.
        target_langs: List of language names. None means all available languages.
        split: Dataset split ('test', 'dev', etc.).
        category: Subject/relation category to filter (required for 'gmmlu' and 'klar').
        max_samples_per_lang: Maximum shared samples per language.
        logger: Optional logger.
        shuffle: Whether to shuffle the DataLoader.

    Returns:
        DataLoader with input_ids, attention_mask, original_texts, labels,
        src_langs, ids, categories, instruction_ids, and question_masks.
    """
    if mode != 'evaluation':
        raise ValueError(f"Multilingual benchmarks only support mode='evaluation', got '{mode}'")
    return _load_multiling_evaluation(
        dataset_name, tokenizer, batch_size, target_langs, split,
        category, max_samples_per_lang, shuffle=shuffle, logger=logger,
    )


def _load_multiling_evaluation(dataset_name, tokenizer, batch_size, target_langs, split,
                                category, max_samples_per_lang, shuffle=False, logger=None):
    """Internal: evaluation mode for load_multiling_dataset."""
    if dataset_name == 'gmmlu':
        assert category is not None, "Please specify category (subject) for GMMLU dataset."
        datasets = []
        if target_langs is None:
            target_langs = list(GMMLU_LANG_CODE_MAP.keys())
        for lang in target_langs:
            lang_code = GMMLU_LANG_CODE_MAP[lang]
            dataset = hf_load_dataset('CohereLabs/Global-MMLU', lang_code)[split]
            if category != 'all':
                dataset = dataset.filter(lambda x: x['subject'] == category)
            else:
                # limit to 30 samples per category for each language
                category_counts = defaultdict(int)
                def category_filter(example):
                    cat = example['subject']
                    if category_counts[cat] < 30:
                        category_counts[cat] += 1
                        return True
                    else:
                        return False
                dataset = dataset.filter(category_filter)
            prompt_templates = GMMLU_PROMPTS[lang]
            for temp_id, prompt_template in enumerate(prompt_templates):
                question_part_template = MMLU_QUESTION_PART[temp_id]
                def preprocess_fn(example):
                    prompt = prompt_template.format(
                        question=example['question'],
                        option_a=str(example['option_a']).strip(),
                        option_b=str(example['option_b']).strip(),
                        option_c=str(example['option_c']).strip(),
                        option_d=str(example['option_d']).strip(),
                    )
                    try:
                        input_text = tokenizer.apply_chat_template(
                            [{'role': 'user', 'content': prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except Exception:
                        input_text = prompt
                    question_part = question_part_template.format(
                        question=example['question'],
                        option_a=str(example['option_a']).strip(),
                        option_b=str(example['option_b']).strip(),
                        option_c=str(example['option_c']).strip(),
                        option_d=str(example['option_d']).strip(),
                    )
                    parts = [question_part]
                    return {
                        'input_text': input_text,
                        'original_text': question_part,
                        'parts': parts,
                        'labels': example['answer'],
                        'language': lang,
                        'instruction_id': temp_id,
                        'id': example['sample_id'],
                        'category': example['subject'],
                    }
                dataset_lang = dataset.map(preprocess_fn, remove_columns=dataset.column_names)
                datasets.append(dataset_lang)
        full_dataset = concatenate_datasets(datasets)

    elif dataset_name == 'klar':
        assert category is not None and category != 'all', "Please specify category for KLAR dataset."
        assert split == 'test', "KLAR dataset only has 'test' split."
        klar_relation = category
        dataset_dir = Path(__file__).resolve().parent.parent.parent / 'KLAR-CLC' / 'klar'
        datasets = []
        if target_langs is None:
            target_langs = list(KLAR_LANG_CODE_MAP.keys())
        for lang in target_langs:
            lang_code = KLAR_LANG_CODE_MAP[lang]
            dataset_path = dataset_dir / lang_code / f'{klar_relation}.json'
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data_json = json.load(f)
            dataset = Dataset.from_list(data_json['samples'])
            if lang != 'English':
                # exclude dataset if 'subject_en' is missing, which indicates low-quality machine translation
                dataset = dataset.filter(lambda x: x['subject_en'] is not None and x['object_en'] is not None)
            prompt_templates = data_json['prompt_templates']
            for temp_id, prompt_template in enumerate(prompt_templates):
                def preprocess_fn(example):
                    prompt = prompt_template.replace('<subject>', example['subject']).replace('<mask>', '').strip()
                    try:
                        input_text = tokenizer.apply_chat_template(
                            [{'role': 'user', 'content': prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except Exception:
                        input_text = prompt
                    return {
                        'input_text': input_text,
                        'original_text': prompt,
                        'parts': [prompt],
                        'labels': example['object'],
                        'language': lang,
                        'instruction_id': temp_id,
                        'id': example['index'],
                        'category': klar_relation,
                    }
                dataset_lang = dataset.map(preprocess_fn, remove_columns=dataset.column_names)
                datasets.append(dataset_lang)
        full_dataset = concatenate_datasets(datasets)

    elif dataset_name == 'belebele':
        assert category is None, "Category filtering for BELEBELE is not supported"
        if target_langs is None:
            target_langs = list(BELEBELE_LANG_CODE_MAP.keys())
        target_langs = sorted(target_langs)

        datasets = []
        for lang in target_langs:
            lang_code = BELEBELE_LANG_CODE_MAP[lang]
            dataset = hf_load_dataset('facebook/belebele', lang_code)[split]
            # id を'link', 'question_number'ごとに付与する
            dataset = dataset.map(lambda x: {'id': f"{x['link']}_{x['question_number']}"})
            prompt_templates = BELEBELE_PROMPTS[lang]

            for temp_id, prompt_template in enumerate(prompt_templates):
                def preprocess_fn(example):
                    prompt = prompt_template.format(
                        passage=example['flores_passage'],
                        question=example['question'],
                        option_a=example['mc_answer1'],
                        option_b=example['mc_answer2'],
                        option_c=example['mc_answer3'],
                        option_d=example['mc_answer4'],
                    )
                    try:
                        input_text = tokenizer.apply_chat_template(
                            [{'role': 'user', 'content': prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except Exception:
                        input_text = prompt
                    answer_map = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
                    return {
                        'input_text': input_text,
                        'original_text': prompt,
                        'parts': [prompt],
                        'labels': answer_map[str(example['correct_answer_num'])],
                        'language': lang,
                        'instruction_id': temp_id,
                        'id': example['id'],
                        'category': 'general',
                    }
                dataset_lang = dataset.map(preprocess_fn, remove_columns=dataset.column_names)
                datasets.append(dataset_lang)
        full_dataset = concatenate_datasets(datasets)

    elif dataset_name == 'xquad':
        assert category is None, "Category filtering for XQUAD is not supported"
        if target_langs is None:
            target_langs = list(XQUAD_LANG_CODE_MAP.keys())
        target_langs = sorted(target_langs)

        datasets = []
        for lang in target_langs:
            lang_code = XQUAD_LANG_CODE_MAP[lang]
            dataset = hf_load_dataset('google/xquad', f'xquad.{lang_code}')[split]
            prompt_templates = XQUAD_PROMPTS[lang]

            for temp_id, prompt_template in enumerate(prompt_templates):
                def preprocess_fn(example):
                    prompt = prompt_template.format(
                        passage=example['context'],
                        question=example['question'],
                    )
                    try:
                        input_text = tokenizer.apply_chat_template(
                            [{'role': 'user', 'content': prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except Exception:
                        input_text = prompt

                    answer = normalize_answer(example['answers']['text'][0], lang=lang) if example['answers']['text'] else ''
                    return {
                        'input_text': input_text,
                        'original_text': prompt,
                        'parts': [prompt],
                        'labels': answer,
                        'language': lang,
                        'instruction_id': temp_id,
                        'id': example['id'],
                        'category': 'qa',
                    }
                dataset_lang = dataset.map(preprocess_fn, remove_columns=dataset.column_names)
                datasets.append(dataset_lang)
        full_dataset = concatenate_datasets(datasets)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    if max_samples_per_lang is not None:
        lang_ids = defaultdict(set)
        for item in full_dataset:
            lang_ids[item['language']].add(f"{item['id']}_{item['instruction_id']}")
        shared_ids = list(set.intersection(*lang_ids.values()))
        shared_ids = sorted(shared_ids)
        random.seed(42)
        selected_ids = random.sample(shared_ids, min(max_samples_per_lang, len(shared_ids)))
        full_dataset = full_dataset.filter(
            lambda x: f"{x['id']}_{x['instruction_id']}" in selected_ids
        )

    full_dataset = full_dataset.sort('id')
    num_samples_per_lang = defaultdict(int)
    for item in full_dataset:
        num_samples_per_lang[item['language']] += 1
    if logger:
        logger.info(f"Loaded {dataset_name} with {dict(num_samples_per_lang)} samples per language.")

    def collate_fn(batch):
        input_texts = [item['input_text'] for item in batch]
        original_texts = [item['original_text'] for item in batch]
        tokenized = tokenizer(input_texts, return_tensors='pt', padding='longest',
                               truncation=False, add_special_tokens=False)
        num_parts = len(batch[0]['parts'])
        question_masks = []
        for i in range(num_parts):
            parts_batch = [item['parts'][i] for item in batch]
            question_mask = _batch_question_masks(tokenizer, input_texts, parts_batch,
                                                  tokenized['input_ids'].shape[1])
            question_masks.append(question_mask)
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': [item['labels'] for item in batch],
            'original_texts': original_texts,
            'src_langs': [item['language'] for item in batch],
            'ids': [item['id'] for item in batch],
            'categories': [item['category'] for item in batch],
            'instruction_ids': [item['instruction_id'] for item in batch],
            'question_masks': question_masks,
        }

    sampler = _make_sampler(full_dataset, shuffle=shuffle)
    return torch.utils.data.DataLoader(
        full_dataset, batch_size=batch_size, shuffle=(sampler is None and shuffle),
        collate_fn=collate_fn, sampler=sampler,
    )


def load_aya_dataset(tokenizer, target_langs=None, split='train', mode='training',
                     batch_size=16, logger=None, max_samples_per_lang=None,
                     max_token_length=None, shuffle=False):
    """Load the CohereLabs/aya_dataset for the paper's SFT objective.

    Args:
        tokenizer: Tokenizer for the model.
        target_langs: List of language names (matching the dataset 'language' field,
                      e.g. 'English', 'French') or None for all languages.
        split: Dataset split (the paper uses 'train').
        mode: Only 'training' is supported.
        batch_size: Batch size for the DataLoader.
        logger: Optional logger.
        max_samples_per_lang: Maximum number of samples per language (randomly sampled
                              with seed=42).
        max_token_length: Maximum token length for truncation.
        shuffle: Whether to shuffle the DataLoader.

    Returns:
        DataLoader with input_ids, attention_mask, labels, question_masks, ids,
        instruction_ids, language codes, language names, and categories.
    """
    ds = hf_load_dataset('CohereLabs/aya_dataset', split=split)

    # Filter by language name if requested
    if target_langs is not None:
        if isinstance(target_langs, str):
            target_langs = [target_langs]
        for lang in target_langs:
            if lang not in AYA_LANG_CODE_MAP:
                raise ValueError(f"Language '{lang}' not found in AYA_LANG_CODE_MAP.")
        target_langs_code_set = set(AYA_LANG_CODE_MAP[lang] for lang in target_langs)
        ds = ds.filter(lambda x: x['language_code'] in target_langs_code_set)
        # exclude 'Traditional Chinese'
        ds = ds.filter(lambda x: x['language'] != 'Traditional Chinese') 

    # Group by language, then sample up to max_samples_per_lang per language
    if max_samples_per_lang is not None:
        lang_indices = defaultdict(list)
        for idx, lang in enumerate(ds['language']):
            lang_indices[lang].append(idx)
        rng = random.Random(42)
        selected_indices = []
        for lang, indices in lang_indices.items():
            selected_indices.extend(rng.sample(indices, min(max_samples_per_lang, len(indices))))
        selected_indices.sort()
        ds = ds.select(selected_indices)

    # Assign a stable integer id using the dataset row index
    ds = ds.map(lambda x, idx: {'id': idx}, with_indices=True)

    if logger:
        item_count = defaultdict(int)
        for lang in ds['language']:
            item_count[lang] += 1
        logger.info(f"Loaded aya_dataset split={split} before preprocessing: {dict(item_count)}")
    if mode == 'training':
        def preprocess_fn(example):
            user_text = example['inputs'].strip()
            assistant_text = example['targets'].strip()
            try:
                instruction_text = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': user_text}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                full_text = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': user_text},
                     {'role': 'assistant', 'content': assistant_text}],
                    tokenize=False, add_generation_prompt=False, enable_thinking=False,
                )
            except Exception:
                instruction_text = user_text
                full_text = user_text + assistant_text
            return {
                'input_text': full_text,
                'instruction_text': instruction_text,
                'original_text': user_text,
                'language': example['language'],
                'lang_code': example['language_code'],
                'instruction_id': 0,
                'id': example['id'],
                'category': example.get('annotation_type', 'aya'),
            }
        full_dataset = ds.map(preprocess_fn, remove_columns=ds.column_names)

        if logger:
            item_count = defaultdict(int)
            for lang in full_dataset['language']:
                item_count[lang] += 1
            logger.info(f"Loaded aya_dataset (training) split={split}: {dict(item_count)}")

        full_dataset = full_dataset.sort('id')

        def collate_fn(batch):
            input_texts = [item['input_text'] for item in batch]
            instruction_texts = [item['instruction_text'] for item in batch]
            original_texts = [item['original_text'] for item in batch]

            tokenized = tokenizer(input_texts, return_tensors='pt', padding='longest',
                                  truncation=max_token_length is not None,
                                  max_length=max_token_length, add_special_tokens=False)

            labels = _batch_labels(tokenizer, input_texts, instruction_texts, tokenized['input_ids'])
            question_mask = _batch_question_masks(tokenizer, input_texts, original_texts,
                                                  tokenized['input_ids'].shape[1])
            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': labels,
                'question_masks': [question_mask],
                'ids': [item['id'] for item in batch],
                'instruction_ids': [item['instruction_id'] for item in batch],
                'languages': [item['lang_code'] for item in batch],
                'src_langs': [item['language'] for item in batch],
                'categories': [item['category'] for item in batch],
            }

    else:
        raise ValueError(f"Aya only supports mode='training', got '{mode}'")

    sampler = _make_sampler(full_dataset, shuffle=shuffle)
    return DataLoader(full_dataset, batch_size=batch_size, shuffle=(sampler is None and shuffle),
                      collate_fn=collate_fn, sampler=sampler)


def load_dataset(dataset_name, tokenizer, mode='embedding', batch_size=16,
                 target_langs=None, split='dev',
                 category=None, max_samples_per_lang=None, logger=None,
                 max_token_length=None, shuffle=False):
    """Unified top-level dataset loader.

    Routes to the appropriate dataset-specific loader based on dataset_name.

    Args:
        dataset_name: One of 'flores_plus', 'ntrex', 'wmtpp', 'gmmlu',
                      'klar', 'belebele', 'xquad', or 'aya_dataset'.
        tokenizer: Tokenizer for the model.
        mode: 'embedding' for hidden-state extraction; 'evaluation' for generation evaluation;
          'training' for instruction-tuning style (returns labels + question_masks).
        batch_size: Batch size for the DataLoader.
        target_langs: List of language names. None means all supported languages.
        split: Dataset split.
        category: Subject/relation filter (required for 'gmmlu' and 'klar').
        max_samples_per_lang: Maximum samples per language.
        logger: Optional logger.
        max_token_length: Maximum token length for truncation (embedding mode only).
        shuffle: Whether to shuffle the DataLoader.

    Returns:
        DataLoader whose collate keys depend on mode (see individual loaders for details).
    """
    if dataset_name == 'flores_plus':
        if mode != 'embedding':
            raise ValueError(f"FLORES+ only supports mode='embedding', got '{mode}'")
        return load_flores_plus(tokenizer, target_langs=target_langs, split=split,
                                batch_size=batch_size, logger=logger,
                                max_samples_per_lang=max_samples_per_lang,
                                max_token_length=max_token_length, shuffle=shuffle)
    elif dataset_name == 'ntrex':
        return load_ntrex(tokenizer, target_langs=target_langs, split=split, mode=mode,
                          batch_size=batch_size, logger=logger,
                          max_samples_per_lang=max_samples_per_lang, max_token_length=max_token_length, shuffle=shuffle)
    elif dataset_name == 'wmtpp':
        return load_wmtpp(tokenizer, target_langs=target_langs, split=split, mode=mode,
                          batch_size=batch_size, logger=logger,
                          max_samples_per_lang=max_samples_per_lang, max_token_length=max_token_length, shuffle=shuffle)
    elif dataset_name in ('gmmlu', 'klar', 'belebele', 'xquad'):
        return load_multiling_dataset(
            dataset_name, tokenizer, mode=mode, batch_size=batch_size,
            target_langs=target_langs, split=split,
            category=category, max_samples_per_lang=max_samples_per_lang,
            logger=logger, shuffle=shuffle,
        )
    elif dataset_name == 'aya_dataset':
        return load_aya_dataset(
            tokenizer, target_langs=target_langs, split=split, mode=mode,
            batch_size=batch_size, logger=logger,
            max_samples_per_lang=max_samples_per_lang,
            max_token_length=max_token_length, shuffle=shuffle,
        )
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")
