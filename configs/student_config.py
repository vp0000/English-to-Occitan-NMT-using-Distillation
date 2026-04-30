from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
import os

@dataclass
class StudentTrainingConfig:
    """Master configuration for student fine-tuning."""

    # Model
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_seq_length: int = 512

    # Training hyperparameters
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    num_train_epochs: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    lr_scheduler_type: str = "cosine"

    # Hardware optimization
    fp16: Optional[bool] = None
    gradient_checkpointing: bool = False

    # Logging and evaluation
    eval_steps: int = 500
    logging_steps: int = 500
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Miscellaneous
    report_to: str = "wandb" # Could be changed to tensorboard for local logging or none if tracking is not desired
    optim: str = "adamw_torch" # Changed to adam because of smaller size and smaller training data
    remove_unused_columns: bool = False
    output_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "tinyllama_distilled")) # Ensure this is defined before use
    run_name: Optional[str] = None if report_to != "wandb" else "training_finetuning"

    # Quantization and LoRA (using actual HF/PEFT classes)
    bnb_config: Optional[BitsAndBytesConfig] = field(default_factory=lambda: BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ))

    lora_config: Optional[LoraConfig] = field(default_factory=lambda: LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules='all-linear',
        lora_dropout=0.15,
        bias="none",
        task_type="CAUSAL_LM"
    ))

    # Data Collator Configuration
    collator_padding: bool = True
    collator_pad_to_multiple_of: Optional[int] = 8
    collator_label_pad_token_id: int = -100

    def __post_init__(self):
        """Validate and set dynamic defaults."""
        if self.fp16 is None:
            self.fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7

        if self.report_to == "stdout":
            print("Warning: 'stdout' not supported, defaulting to 'none'")
            self.report_to = "none"

    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            fp16=self.fp16,
            gradient_checkpointing=self.gradient_checkpointing,
            logging_steps=self.logging_steps,
            eval_strategy=self.eval_strategy,
            save_strategy=self.save_strategy,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            report_to=self.report_to if self.report_to != "stdout" else "none",
            optim=self.optim,
            remove_unused_columns=self.remove_unused_columns,
            run_name=self.run_name
        )