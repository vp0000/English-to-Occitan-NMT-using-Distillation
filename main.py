from transformers import AutoTokenizer, MarianTokenizer, MarianMTModel
from datasets import Dataset
import json
import os
import pandas as pd
import wandb
from typing import Tuple, Optional
from time import time
from dotenv import load_dotenv

from configs.base_config import Config
from data.download_data import download_corpus
from data.preprocessing import preprocess_corpus
from data.sampling import perform_sampling
from utils.tokenizer import tokenize_data
from configs.teacher_config import TeacherTrainingConfig
from configs.student_config import StudentTrainingConfig
from training.teacher_training import train_teacher_model
from training.teacher_generate import generate_translations
from training.student_training import train_student_model
from utils.load_model import load_saved_model
from evaluation.eval_translation import generate_eval_metrics
from data.eval_create import create_parallel_dataset

def model_name_extract(model: str) -> str: return model.split('/')[0]

def main():
    # Step 1: Download and preprocess the corpus
    base_config = Config()
    raw_corpus_path = download_corpus(
        base_config.url,
        base_config.dataset_name,
        base_config.lang_codes[0],
        base_config.lang_codes[1],
        base_config.base_dir,
        base_config.file_ext
    )
    preprocessed_corpus= preprocess_corpus(raw_corpus_path, base_config.dataset_name, base_config.lang_codes)
    
    # Step 1.5: Clean the training dataset

    # Step 2: Sample the corpus for training
    assert preprocessed_corpus is not None, "Preprocessed corpus is None. Check the preprocessing step."
    data_tuple = perform_sampling(
        preprocessed_corpus,
        base_config.train_size,
        base_config.train_earlystop_size,
        base_config.distill_size,
        base_config.distill_earlystop_size,
        base_config.sampling_strategy,
        base_config.random_seed
    )
    train_data, train_earlystop_data, distill_data, distill_earlystop_data = data_tuple

    # Step 3: Load teacher tokenizer and prepare training datasets
    teacher_token_dir = f"{base_config.base_dir}/teacher_tokenizer"
    train_dataset = tokenize_data(
        train_data,
        base_config.teacher_model,
        base_config.lang_codes,
        teacher_token_dir
    )
    eval_dataset = tokenize_data(
        train_earlystop_data,
        base_config.teacher_model,
        base_config.lang_codes,
        teacher_token_dir
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_token_dir)  # Reload tokenizer from the saved directory

    # Step 4: Train the teacher model
    curr_time = int(time())
    if base_config.report == "wandb":
        load_dotenv()
        wandb.init(project=os.getenv('WANDB_PROJECT_NAME'))
    run_name = f"{model_name_extract(base_config.teacher_model)}_size={len(train_data)}_clean={base_config.cleaning_flag}_sampling={base_config.sampling_strategy}_{curr_time}"
    teacher_config = TeacherTrainingConfig(run_name=run_name, output_dir=os.path.join(base_config.base_dir, "mistral_occitan_finetuned"))
    output_dir_t = f"{teacher_config.output_dir}/final"
    teacher_trainer = train_teacher_model(teacher_config, teacher_tokenizer, train_dataset, eval_dataset, output_dir_t)
    wandb.finish()
    
    # Step 5: Generate translations using the teacher model
    teacher_model = load_saved_model(base_config.teacher_model, output_dir_t, teacher_config)
    distill_translations = generate_translations(
        model=teacher_model,
        tokenizer=teacher_tokenizer,
        distill_data=distill_data,
        input_column=base_config.lang_codes[0],
        target_column=base_config.lang_codes[0],
        batch_size=50
    )

    # Step 6: Load student tokenizer and prepare distillation datasets
    student_token_dir = f"{base_config.base_dir}/student_tokenizer"
    distill_dataset = tokenize_data(
        distill_translations,
        base_config.student_model,
        base_config.lang_codes,
        student_token_dir
    )
    distill_eval_dataset = tokenize_data(
        distill_earlystop_data,
        base_config.student_model,
        base_config.lang_codes,
        student_token_dir
    )
    student_tokenizer = AutoTokenizer.from_pretrained(student_token_dir)  # Reload tokenizer from the saved directory

    # Step 7: Train the student model
    curr_time = int(time())
    if base_config.report == "wandb":
        wandb.init(project=os.getenv('WANDB_PROJECT_NAME'))
    run_name = f"{model_name_extract(base_config.student_model)}_size={len(train_data)}_clean={base_config.cleaning_flag}_sampling={base_config.sampling_strategy}_{curr_time}"
    student_config = StudentTrainingConfig(run_name=run_name, output_dir=os.path.join(base_config.base_dir, "tinyllama_distilled"))
    output_dir_s = f"{student_config.output_dir}/final"
    student_trainer = train_student_model(student_config, student_tokenizer, distill_dataset, distill_eval_dataset, output_dir_s)
    wandb.finish()

    # Step 8: Create final eval dataset and generate metrics
    eval_parallel = create_parallel_dataset(base_config.flores_dir, base_config.flores_codes, base_config.lang_codes, 'dev')
    marian_tokenizer = MarianTokenizer.from_pretrained(base_config.eval_model)
    marian_model = MarianMTModel.from_pretrained(base_config.eval_model)
    student_model = load_saved_model(base_config.student_model, output_dir_s, student_config)
    # Teacher (causal)
    teacher_metrics = generate_eval_metrics(teacher_model, eval_parallel, teacher_tokenizer, batch_size=10, model_type="causal")
    # Student (causal)
    student_metrics = generate_eval_metrics(student_model, eval_parallel, student_tokenizer, batch_size=10, model_type="causal")
    # Baseline (Marian)
    baseline_metrics = generate_eval_metrics(marian_model, eval_parallel, marian_tokenizer, batch_size=10, model_type="seq2seq", lang_code="oci")

    # Step 9: Print metrics
    metrics_keys = ['Teacher', 'Student', 'Baseline']
    metrics_dicts = [teacher_metrics, student_metrics, baseline_metrics]
    for i in range(3):
        print(metrics_keys[i])
        print(" ".join(f"{key} -> {value}" for key, value in metrics_dicts[i].items()))

if __name__ == "__main__": main()