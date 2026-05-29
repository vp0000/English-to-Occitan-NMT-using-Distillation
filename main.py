from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, MarianMTModel, MarianTokenizer
from huggingface_hub import login
from time import time, strftime, localtime
from typing import Optional
from dotenv import load_dotenv
import os
import wandb
import torch
import pandas as pd
import gc

from configs.base_config import Config
from configs.cleaning_config import Cleaning_Config
from utils.setup_langid_model import setup_langid_model
from data.download_data import download_corpus
from data.preprocessing import preprocess_corpus
from data.sampling import perform_sampling
from data.cleaning import cleaning_orchestrator
from utils.tokenizer import tokenize_data
from configs.teacher_config import TeacherTrainingConfig
from configs.student_config import StudentTrainingConfig
from training.teacher_training import train_teacher_model
from training.teacher_generate import generate_translations
from training.student_training import train_student_model
from utils.load_model import load_saved_model
from utils.checkpoint_handling import check_checkpoint, model_exists
from evaluation.eval_translation import generate_eval_metrics
from data.eval_create import create_parallel_dataset

# from transformers.utils import logging

# # Set the verbosity to ERROR level (int value 40)
# logging.set_verbosity_error()

def model_name_extract(model: str) -> str: return model.split('/')[-1]
def cached_or_compute(path: Optional[str], compute_fn):
    """Read file from path if it exists, otherwise run compute_fn and (optionally) save."""
    if path:
        # print(path)
        root, extension = os.path.splitext(path)
        if os.path.exists(path):
            if extension.lower() == ".xlsx":
                return pd.read_excel(path)
            elif extension.lower() == ".csv":
                return pd.read_csv(path)
            else:
                raise ValueError('Filepath should have an xlsx or a csv extension.')
        else:
            df = compute_fn() #Throwaway args variable
            os.makedirs(os.path.dirname(root), exist_ok=True)
            # print(extension)
            if extension.lower() == ".xlsx":
                df = df.replace(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', regex=True) # Remove invalid characters for Excel, csv can handle them so not needed there
                df.to_excel(path, index=False)
            elif extension.lower() == ".csv":
                df.to_csv(path, index=False, encoding="utf-8-sig")
            else:
                raise ValueError('Filepath should have an xlsx or a csv extension.')
            return df
    else:
        raise ValueError('Filepath must be a string. It cannot be none.')

def main():
    load_dotenv()
    login(os.environ.get('HF_TOKEN'))
    cleaning_flag = os.path.exists(os.path.join(os.getcwd(), 'data/cleaning.py'))
    # Step 1: Download and preprocess the corpus
    base_config = Config(cleaning_flag=cleaning_flag)
    preprocessed_output_path = os.path.join(base_config.base_dir, "datasets/preprocessed_data.csv")
    # print(preprocessed_output_path)
    preprocessed_corpus = cached_or_compute(preprocessed_output_path,
    lambda: preprocess_corpus(download_corpus(base_config.url,
            base_config.dataset_name,
            base_config.lang_codes[0],
            base_config.lang_codes[1],
            base_config.base_dir,
            base_config.file_ext), base_config.dataset_name, base_config.lang_codes)
)
    
    # Step 1.5: Clean the training dataset
    cleaning_config = Cleaning_Config()
    if base_config.cleaning_flag:
        assert preprocessed_corpus is not None, "Preprocessed corpus is None. Check the preprocessing step."
        langid_model_path = setup_langid_model()
        output_path = os.path.join(base_config.base_dir, "datasets/cleaned_data.csv")
        cleaned_corpus = cached_or_compute(
            output_path,
            lambda: cleaning_orchestrator(
            df = preprocessed_corpus,
            lang_map = cleaning_config.lang_map,
            confidence_dict = cleaning_config.confidence_dict,
            src_col = base_config.lang_codes[0],
            tgt_col = base_config.lang_codes[1],
            punct_threshold = cleaning_config.punct_threshold,
            len_thres = cleaning_config.len_thres,
            word_thres = cleaning_config.word_thres,
            lower_mult_thres = cleaning_config.lower_mult_thres,
            upper_mult_thres = cleaning_config.upper_mult_thres,
            label_col = cleaning_config.label_col,
            lang_id_model_path = langid_model_path,
            embed_model_name = cleaning_config.embed_model_name,
            chunk_size = cleaning_config.chunk_size,
            quantile_thres = cleaning_config.quantile_thres,
            output_path = output_path
        )
    )
    else:
        cleaned_corpus = preprocessed_corpus

    # Step 2: Sample the corpus for training
    if cleaned_corpus is None:
        score_flag = False
        assert preprocessed_corpus is not None, "Preprocessed corpus is None. Check the preprocessing step."
        data_tuple = perform_sampling(
            df = preprocessed_corpus,
            text_cols = base_config.lang_codes,
            score_col = base_config.score_column,
            train_size = base_config.train_size,
            distill_size = base_config.distill_size,
            train_earlystop_size = base_config.train_earlystop_size,
            distill_earlystop_size = base_config.distill_earlystop_size,
            sampling_strategy = base_config.sampling_strategy,
            random_seed = base_config.random_seed,
            score_flag = score_flag
        )
        train_data, distill_data, train_earlystop_data,  distill_earlystop_data = data_tuple
    else:
        score_flag = True
        data_tuple = perform_sampling(
            df = cleaned_corpus,
            text_cols = base_config.lang_codes,
            score_col = base_config.score_column,
            train_size = base_config.train_size,
            distill_size = base_config.distill_size,
            train_earlystop_size = base_config.train_earlystop_size,
            distill_earlystop_size = base_config.distill_earlystop_size,
            sampling_strategy = base_config.sampling_strategy,
            random_seed = base_config.random_seed,
            score_flag = score_flag
        )
        train_data, distill_data, train_earlystop_data,  distill_earlystop_data = data_tuple

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
    curr_time = strftime("%Y-%m-%d %H:%M:%S", localtime(time()))
    teacher_config = TeacherTrainingConfig(output_dir=os.path.join(base_config.base_dir, "mistral_occitan_finetuned"), report_to=base_config.report)
    output_dir_t = f"{teacher_config.output_dir}/final"
    # print(teacher_config.output_dir)
    if model_exists(output_dir_t):
        print(f"Teacher model found at {output_dir_t}, skipping training.")
    else:
        checkpoint = check_checkpoint(base_config.checkpoints, teacher_config.output_dir)
        if base_config.report == "wandb":
            wandb.login(key=os.environ.get('WANDB_API_KEY')) # type: ignore[attr-defined]
            run_name = f"{model_name_extract(base_config.teacher_model)}_size={len(train_data)}_clean={base_config.cleaning_flag}_sampling={base_config.sampling_strategy}_{curr_time}"
            wandb.init(project=os.environ.get('WANDB_PROJECT'), name=run_name) # type: ignore[attr-defined]
            train_teacher_model(teacher_config, teacher_tokenizer, train_dataset, eval_dataset, checkpoint, output_dir_t)
            wandb.finish() # type: ignore[attr-defined]
        else:
            train_teacher_model(teacher_config, teacher_tokenizer, train_dataset, eval_dataset, checkpoint, output_dir_t)
    
    # Step 5: Generate translations using the teacher model, followed by tokenization and student training
    student_config = StudentTrainingConfig(output_dir=os.path.join(base_config.base_dir, "tinyllama_distilled"), report_to=base_config.report)
    student_token_dir = f"{base_config.base_dir}/student_tokenizer"
    output_dir_s = f"{student_config.output_dir}/final"
    if model_exists(output_dir_s):
        student_tokenizer = AutoTokenizer.from_pretrained(student_token_dir)
        print(f"Student model found at {output_dir_s}, skipping training.")
    else:
        teacher_model = load_saved_model(output_dir_t)
        output_path = os.path.join(base_config.base_dir, "datasets/distillation_data.csv")
        distill_translations = cached_or_compute(
            output_path,
            lambda: generate_translations(
            model=teacher_model,
            tokenizer=teacher_tokenizer,
            distill_data=distill_data,
            output_path=output_path,
            input_column=base_config.lang_codes[0],
            target_column=base_config.lang_codes[1],
            batch_size=64
        )
    )
        del teacher_model
        gc.collect()
        torch.cuda.empty_cache()
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
        student_tokenizer = AutoTokenizer.from_pretrained(student_token_dir)
        curr_time = strftime("%Y-%m-%d %H:%M:%S", localtime(time()))
        checkpoint = check_checkpoint(base_config.checkpoints, student_config.output_dir)
        if base_config.report == "wandb":
            run_name = f"{model_name_extract(base_config.student_model)}_size={len(train_data)}_clean={base_config.cleaning_flag}_sampling={base_config.sampling_strategy}_{curr_time}"
            wandb.init(project=os.environ.get('WANDB_PROJECT'), name=run_name)  # type: ignore[attr-defined]
            train_student_model(student_config, student_tokenizer, distill_dataset, distill_eval_dataset, checkpoint, output_dir_s)
            wandb.finish() # type: ignore[attr-defined]
        else:
            train_student_model(student_config, student_tokenizer, distill_dataset, distill_eval_dataset, checkpoint, output_dir_s)
    
    # Step 6: Create final eval dataset and generate metrics
    eval_parallel = create_parallel_dataset(base_config.flores_dir, base_config.flores_codes, base_config.lang_codes, 'dev')
    # nllb_tokenizer = AutoTokenizer.from_pretrained(base_config.eval_model)
    # nllb_model = AutoModelForSeq2SeqLM.from_pretrained(base_config.eval_model)
    marian_model = MarianMTModel.from_pretrained(base_config.marian_model)
    marian_tokenizer = MarianTokenizer.from_pretrained(base_config.marian_model)
    nllb_flag = False
    
    teacher_model = load_saved_model(output_dir_t)
    student_model = load_saved_model(output_dir_s)
    teacher_outputs_path = os.path.join(base_config.base_dir, "datasets/teacher_eval.csv")
    student_outputs_path = os.path.join(base_config.base_dir, "datasets/student_eval.csv")
    baseline_outputs_path = os.path.join(base_config.base_dir, "datasets/baseline_eval.csv")
    
    # Teacher (causal)
    teacher_metrics = generate_eval_metrics(
        teacher_model,
        eval_parallel,
        teacher_tokenizer,
        batch_size=20,
        model_type="causal",
        output_path=teacher_outputs_path
    )
    del teacher_model
    gc.collect()
    torch.cuda.empty_cache()

    # Student (causal)
    student_metrics = generate_eval_metrics(
        student_model,
        eval_parallel,
        student_tokenizer,
        batch_size=20,
        model_type="causal",
        output_path=student_outputs_path
    )
    del student_model
    gc.collect()
    torch.cuda.empty_cache()

    # Baseline (NLLB)
    # Wrap this in the cached_or_compute wrapper as well
    baseline_metrics = generate_eval_metrics(
        marian_model,
        eval_parallel,
        marian_tokenizer,
        batch_size=20,
        model_type="seq2seq",
        lang_code="oci",
        src_lang="eng_Latn",
        tgt_lang="oci_Latn",
        nllb_flag=nllb_flag,
        num_beams=5,
        output_path=baseline_outputs_path
    )
    del marian_model
    gc.collect()
    torch.cuda.empty_cache()

    # Zero-Shot Baseline
    zs_teacher_model, _ = FastLanguageModel.from_pretrained(
        base_config.teacher_model,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True
    )
    zs_teacher_tokenizer = AutoTokenizer.from_pretrained("base_config.teacher_model")
    zs_teacher_outputs_path = os.path.join(base_config.base_dir, "datasets/zs_teacher_eval.csv")
    zs_teacher_metrics = generate_eval_metrics(
            zs_teacher_model,
            eval_parallel,
            zs_teacher_tokenizer,
            batch_size=20,
            model_type="causal",
            output_path=zs_teacher_outputs_path
        )
    del zs_teacher_model
    gc.collect()
    torch.cuda.empty_cache()

    # Zero-Shot Student Model
    zs_student_model, _ = FastLanguageModel.from_pretrained(
        base_config.student_model,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True
    )
    zs_student_tokenizer = AutoTokenizer.from_pretrained(base_config.student_model)
    zs_student_outputs_path = os.path.join(base_config.base_dir, "datasets/zs_student_eval.csv")
    zs_student_metrics = generate_eval_metrics(
            zs_student_model,
            eval_parallel,
            zs_student_tokenizer,
            batch_size=20,
            model_type="causal",
            output_path=zs_student_outputs_path
        )
    del zs_student_model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 7: Print metrics
    metrics_keys = ['Teacher', 'Student', 'Baseline(Marian)', 'Zero-Shot Teacher', 'Zero-Shot Student']
    metrics_dicts = [teacher_metrics, student_metrics, baseline_metrics, zs_teacher_metrics, zs_student_metrics]
    for i in range(5):
        print(metrics_keys[i])
        print(" ".join(f"{key} -> {value}" for key, value in metrics_dicts[i].items()))

if __name__ == "__main__": main()