from unsloth import FastLanguageModel

def load_saved_model(saved_dir: str, max_seq_length: int = 512):
    model, _ = FastLanguageModel.from_pretrained(
        model_name=saved_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto-detect bf16/fp16
    )
    FastLanguageModel.for_inference(model)
    return model