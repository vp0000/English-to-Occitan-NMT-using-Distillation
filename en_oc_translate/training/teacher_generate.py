from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
from peft import PeftModel
from configs.teacher_config import TeacherTrainingConfig
from sacrebleu import corpus_bleu
import os
import json
import torch
import pandas as pd
from tqdm import tqdm

def initialize_pipeline(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto"):
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=dtype,
        device_map=device_map)

def batch_generate(
        sentences: pd.Series,
        pipeline: pipeline,
        # generation_strat: str = "sequence_kd",
        batch_size: int = 20
    ):
        logging.set_verbosity_error() # Used because max_length is still referenced in the default params causing tqdm to spill in newline for every sentence.
        all_sentences = sentences.tolist()
        translations = []

        pipeline.tokenizer.padding_side = "left"

        for i in tqdm(range(0, len(all_sentences), batch_size), desc="Generating"):
            batch = all_sentences[i : i + batch_size]

            # if generation_strat == "sequence_kd":
            # Uses the sequence distillation approach from Kim et. al.(2016) by selecting the most probable sentence generated for an input
            prompts = [f"Translate this English sentence to Occitan: {sentence} ->" for sentence in batch]
            results = pipeline(
                prompts,
                max_new_tokens=256,
                do_sample=False,
                num_beams=5,
                early_stopping=True,
                num_return_sequences=1,
                eos_token_id=pipeline.tokenizer.eos_token_id,
                pad_token_id=pipeline.tokenizer.pad_token_id,
                batch_size=batch_size,
            )
            for j, result in enumerate(results):
                full_text = result[0]['generated_text']
                prompt_len = len(prompts[j])
                generated_part = full_text[prompt_len:].split("\n")[0].strip()
                translations.append(generated_part)

        return pd.Series(translations, index=sentences.index)

def generate_translations(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        distill_data: pd.DataFrame,
        dtype: torch.dtype = torch.float16,
        input_column: str = 'en',
        target_column: str = 'oc',
        batch_size: int = 20
    ):
    gen_pipeline = initialize_pipeline(model, tokenizer, dtype, device_map="auto")
    distillation_data = distill_data.copy(deep=True)
    distillation_data[f'{target_column}'] = distillation_data[f'{input_column}'].pipe(
        batch_generate,
        gen_pipeline,
        batch_size=batch_size
    )
    return distillation_data