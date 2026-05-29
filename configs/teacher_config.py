from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from peft import LoraConfig, get_peft_model
from base_config import Config
import torch
import os

base_vals = Config()
@dataclass
class TeacherTrainingConfig:
    """Master configuration for teacher fine-tuning."""

    # Model
    model_name: str = base_vals.teacher_model
    max_seq_length: int = 512

    # Training hyperparameters
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # Hardware optimization
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    gradient_checkpointing: bool = True

    # Logging and evaluation
    eval_steps: int = 100
    logging_steps: int = 100
    save_steps: int = 100
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Miscellaneous
    report_to: str = "none" # Could be changed to tensorboard for local logging or none if tracking is not desired
    optim: str = "paged_adamw_8bit"
    remove_unused_columns: bool = False
    output_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "mistral_occitan_finetuned")) # Ensure this is defined before use
    attn_implementation = "sdpa" # Default scaled dot product attention
    
    # Quantization and LoRA (using actual HF/PEFT classes)
    bnb_config: Optional[BitsAndBytesConfig] = field(default_factory=lambda: BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ))

    lora_config: Optional[LoraConfig] = field(default_factory=lambda: LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules='all-linear',
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    ))

    # Data Collator Configuration
    collator_padding: bool = True
    collator_pad_to_multiple_of: Optional[int] = 8
    collator_label_pad_token_id: int = -100

    def __post_init__(self):
        """Validate and set dynamic defaults."""
        if torch.cuda.is_available():
            comp, _ = torch.cuda.get_device_capability()
            if comp >= 8:
                self.bf16, self.fp16 = True, False
                self.attn_implementation = "flash_attention_2"
            else:
                self.bf16, self.fp16 = False, True
                self.bnb_config.bnb_4bit_compute_dtype = torch.float16

        if self.report_to == "stdout":
            print("Warning: 'stdout' not supported, defaulting to 'none'")
            self.report_to = "none"
        
        try:
            import flash_attn
            if flash_attn.__version__ >= "2.0.0" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                self.attn_implementation = "flash_attention_2"
        except ImportError:
            pass

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
            bf16=self.bf16,
            gradient_checkpointing=self.gradient_checkpointing,
            logging_steps=self.logging_steps,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            eval_strategy=self.eval_strategy,
            save_strategy=self.save_strategy,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            report_to=self.report_to if self.report_to != "stdout" else "none",
            optim=self.optim,
            remove_unused_columns=self.remove_unused_columns
        )