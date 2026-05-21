from unsloth import FastLanguageModel
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
# from peft import PeftModel
# from configs.teacher_config import TeacherTrainingConfig
# from sacrebleu import corpus_bleu
# import os
# import json
import torch
import pandas as pd
from tqdm import tqdm
import warnings
import regex as re
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def is_good_synthetic(en: str, oc: str) -> bool:
    en, oc = en.strip(), oc.strip()
    if en.lower() == oc.lower():  # copy-through
        return False
    en_w, oc_w = len(en.split()), len(oc.split())
    if oc_w < 2 and en_w > 4:  # collapsed output
        return False
    if max(en_w / max(oc_w, 1), oc_w / max(en_w, 1)) > 3:  # extreme ratio
        return False
    # Check for repetition pattern (1-5 char substring repeated 8+ times)
    for n in range(1, 6):
        if re.search(r'(.{' + str(n) + r'})\1{7,}', oc):
            return False
    return True

def generate_translations(
    model,
    tokenizer,
    distill_data: pd.DataFrame,
    output_path: str,
    input_column: str = "en",
    target_column: str = "oc",
    batch_size: int = 64,
    max_new_tokens: int = 128,
    num_beams: int = 1,
):
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    distillation_data = distill_data.copy(deep=True)
    sentences = distillation_data[input_column]

    # Sort by character length to minimize padding waste within batches
    sorted_index = sentences.str.len().sort_values().index
    sentences_sorted = sentences.loc[sorted_index].tolist()

    translations = []
    for i in tqdm(range(0, len(sentences_sorted), batch_size), desc="Generating"):
        batch = sentences_sorted[i : i + batch_size]
        prompts = [f"Translate the following English sentence to Occitan: {s} ->" for s in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                num_beams = num_beams,
                do_sample = False,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id,
                use_cache = True,
                no_repeat_ngram_size=3,  # Heuristics for now, can be sent to a config file later
                repetition_penalty=1.2,
                length_penalty=1.0,
            )

        gen_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        translations.extend(d.strip() for d in decoded)

    # Restore original order before assigning back
    translations_series = pd.Series(translations, index=sorted_index).sort_index()
    distillation_data[target_column] = translations_series
    # distillation_data['good_synth'] = distillation_data.apply(
    #     lambda row: is_good_synthetic(row[input_column], row[target_column]), axis=1)

    assert output_path is not None
    if 'xlsx' in output_path:
        distillation_data.to_excel(output_path, index=False)
    elif 'csv' in output_path:
        distillation_data.to_csv(output_path, index=False)

    return distillation_data