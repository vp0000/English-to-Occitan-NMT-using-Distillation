import os
import wandb
import json
import torch
from typing import Optional, Tuple
from datasets import Dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from peft import LoraConfig, get_peft_model
from configs.teacher_config import TeacherTrainingConfig

def load_model_with_args(config: TeacherTrainingConfig) -> Tuple[AutoModelForCausalLM, TrainingArguments]:
    """
    Load the base model with quantization and apply LoRA for fine-tuning, then convert the training configuration to HuggingFace TrainingArguments.
    """
    model = AutoModelForCausalLM.from_pretrained(config.model_name, quantization_config=config.bnb_config, device_map="auto")
    teacher_model = get_peft_model(model, peft_config=config.lora_config)
    training_args = config.to_training_arguments()
    return teacher_model,training_args

def define_data_collator(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, config: TeacherTrainingConfig) -> DataCollatorForSeq2Seq:
    """
    Define the data collator for sequence-to-sequence training with given args.
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=config.collator_padding,
        pad_to_multiple_of=config.collator_pad_to_multiple_of,
        label_pad_token_id=config.collator_label_pad_token_id
    )

def dir_check(dir_path: Optional[str]) -> str:
    """
    Ensure the output directory exists before training. If dir_path is None, use a default directory. If the directory does not exist, create it.
    """
    if dir_path is None:
        print("Warning: output_dir is None, using default 'final_model'")
        dir_path = "final_model"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def train_model_with_tracking(
        training_args: TrainingArguments,
        model: AutoModelForCausalLM,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        collator: DataCollatorForSeq2Seq) -> Trainer: # Define training scheme and execute training
    """
    Train the model using HuggingFace Trainer with the specified training arguments, datasets, and data collator.
    Returns the Trainer object after training is complete.
    """
    training_args.output_dir = dir_check(training_args.output_dir) # Ensure output directory exists before training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator
    )
    trainer.train()
    return trainer

def save_model_config(trainer: Trainer, config: TeacherTrainingConfig,dir_path: Optional[str]):
    """
    Save the trained model and its configuration to the specified directory.
    """
    def to_dict(obj) -> dict:
        """Converts the config into a JSON-serializable dictionary."""
        # Create a shallow copy of the dataclass dictionary
        config_dict = obj.__dict__.copy()
        # 1. Handle BitsAndBytesConfig
        if obj.bnb_config is not None:
            bnb_dict = obj.bnb_config.to_dict()           
            # Convert torch.dtype to string (e.g., "torch.bfloat16")
            dtype = bnb_dict.get("bnb_4bit_compute_dtype")
            if isinstance(dtype, torch.dtype):
                bnb_dict["bnb_4bit_compute_dtype"] = str(dtype)                
            config_dict["bnb_config"] = bnb_dict
        # 2. Handle LoraConfig
        if obj.lora_config is not None:
            config_dict["lora_config"] = obj.lora_config.to_dict()
        return config_dict
    dir_path = dir_check(dir_path) # Ensure output directory exists before saving
    trainer.save_model(dir_path)
    config_path = os.path.join(dir_path, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(to_dict(config), f, indent=4)

def train_teacher_model(config: TeacherTrainingConfig, tokenizer: AutoTokenizer, train_dataset: Dataset, eval_dataset: Dataset, dir_path: Optional[str] = None) -> Trainer:
    """
    Main function to train the teacher model. It loads the model and training arguments, defines the data collator, and executes training with tracking.
    Saves the trained model to the specified output directory.
    """
    model, training_args = load_model_with_args(config)
    collator = define_data_collator(tokenizer, model, config)
    trainer = train_model_with_tracking(training_args, model, train_dataset, eval_dataset, collator)
    save_model_config(trainer, config, dir_path)
    return trainer