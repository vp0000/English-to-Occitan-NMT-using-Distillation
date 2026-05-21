from unsloth import FastLanguageModel
import os
import wandb
import json
import torch
from typing import Optional, Tuple
from datasets import Dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from configs.student_config import StudentTrainingConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# def load_model_with_args(config: TeacherTrainingConfig) -> Tuple[AutoModelForCausalLM, TrainingArguments]:
#     """
#     Load the base model with quantization and apply LoRA for fine-tuning, then convert the training configuration to HuggingFace TrainingArguments.
#     """
#     model = AutoModelForCausalLM.from_pretrained(
#         config.model_name,
#         quantization_config = config.bnb_config,
#         device_map = "auto",
#         attn_implementation = config.attn_implementation)
#     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
#     teacher_model = get_peft_model(model, peft_config=config.lora_config)
#     training_args = config.to_training_arguments()
#     return teacher_model,training_args

def load_model_with_args(config: StudentTrainingConfig) -> Tuple[AutoModelForCausalLM, TrainingArguments]:
    """
    Load the base model with quantization and apply LoRA for fine-tuning, then convert the training configuration to HuggingFace TrainingArguments.
    """
    assert config.lora_config is not None
    model, _ = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,                  # auto-detect bf16/fp16
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_config.r,
        lora_alpha=config.lora_config.lora_alpha,
        target_modules=list(config.lora_config.target_modules) if isinstance(
            config.lora_config.target_modules, (list, tuple)
        ) else ["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=config.lora_config.lora_dropout,
        bias=config.lora_config.bias,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    training_args = config.to_training_arguments()
    return model, training_args

def define_data_collator(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, config: StudentTrainingConfig) -> DataCollatorForSeq2Seq:
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
        collator: DataCollatorForSeq2Seq,
        checkpoint: Optional[str|bool] = None) -> Trainer: # Define training scheme and execute training
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

def save_model_config(trainer, config, dir_path=None):
    """Save the trained model and its configuration to the specified directory."""
    
    def make_serializable(obj):
        """Recursively convert an object to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, torch.dtype):
            return str(obj)
        elif hasattr(obj, 'to_dict'):
            # HuggingFace config objects
            return make_serializable(obj.to_dict())
        elif hasattr(obj, '__dict__'):
            # Dataclass or other objects
            return make_serializable(obj.__dict__)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    dir_path = dir_check(dir_path)
    trainer.save_model(dir_path)
    
    config_dict = make_serializable(config.__dict__)
    
    config_path = os.path.join(dir_path, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def train_student_model(
        config: StudentTrainingConfig,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        checkpoint: Optional[str|bool] = None,
        config_path: Optional[str] = None):
    """
    Main function to train the teacher model. It loads the model and training arguments, defines the data collator, and executes training with tracking.
    Saves the trained model to the specified output directory.
    """
    model, training_args = load_model_with_args(config)
    collator = define_data_collator(tokenizer, model, config)
    trainer = train_model_with_tracking(training_args, model, train_dataset, eval_dataset, collator, checkpoint)
    save_model_config(trainer, config, config_path)