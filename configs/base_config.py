from dataclasses import dataclass

@dataclass
class Config:
    """Default configuration for experiments."""
    base_dir: str = "/content/drive/MyDrive/English_to_Occitan_Translation"
    url: str = "https://opus.nlpl.eu/opusapi?"
    token: str = "hug_face_token"
    dataset_name: str = "CCMatrix"
    lang_codes: list = ['en', 'oc']
    file_ext: str = "moses"
    report: str = "wandb"
    sampling_strategy: str = "length_stratified"
    teacher_model: str = "mistralai/Mistral-7B-v0.3"
    student_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    eval_model: str = "Helsinki-NLP/opus-mt-tc-big-en-cat_oci_spa" # Marian model for evaluation
    train_size: int = 50000
    train_earlystop_size: int = 5000
    distill_size: int = 10000
    distill_earlystop_size: int = 1000
    random_seed: int = 42
    cleaning_flag: bool = False  # Set to True if you have cleaning steps to perform before training, check using presence of cleaning.py files in utils
    flores_dir: str = "openlanguagedata/flores_plus"
    flores_codes: list = ['eng_Latn', 'oci_Latn']