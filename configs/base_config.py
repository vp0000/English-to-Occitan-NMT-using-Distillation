from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class Config:
    """Default configuration for experiments."""
    base_dir: str = "/content/drive/MyDrive/English_to_Occitan_Translation_v3.1"
    url: str = "https://opus.nlpl.eu/opusapi?"
    token: str = "hug_face_token"
    dataset_name: str = "CCMatrix"
    lang_codes: list = field(default_factory=lambda: ['en', 'oc'])
    score_column: str = 'final_score'
    file_ext: str = "moses"
    # preprocessed_output_path: str = field(default_factory=lambda: os.path.join(os.getcwd(), "datasets/preprocessed_data.csv"))
    cleaning_flag: bool = True
    report: str = "wandb"
    sampling_strategy: str = "hybrid"
    teacher_model: str = "mistralai/Mistral-7B-v0.3"
    student_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    checkpoints: Optional[str|bool] = True
    eval_model: str = "facebook/nllb-200-3.3B" # NLLB model for evaluation
    train_size: int = 10000
    train_earlystop_size: int = 1000
    distill_size: int = 50000
    distill_earlystop_size: int = 5000
    random_seed: int = 42
    cleaning_flag: bool = False  # Set to True if you have cleaning steps to perform before training, check using presence of cleaning.py files in utils
    flores_dir: str = "openlanguagedata/flores_plus"
    flores_codes: list = field(default_factory=lambda: ['eng_Latn', 'oci_Latn'])
