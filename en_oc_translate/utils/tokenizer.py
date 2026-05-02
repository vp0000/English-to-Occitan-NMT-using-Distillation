import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

def define_tokenizer(model_name: str, is_causal: bool) -> AutoTokenizer:
    assert model_name is not None and isinstance(model_name, str)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if is_causal:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def tokenize_fn(input_data: dict, tokenizer_class: AutoTokenizer, cols: list, max_length: int = 256) -> dict:
    assert input_data is not None
    assert tokenizer_class is not None
    assert cols is not None and all(isinstance(col, str) for col in cols)

    init_list = input_data[cols[0]]
    target_list = input_data[cols[1]]
    assert len(init_list) == len(target_list)

    prompt_list = [f"Translate the following English sentence to Occitan: {text} -> " for text in init_list]
    output_list = [f"{target}{tokenizer_class.eos_token}" for target in target_list]
    prompt_tokens = tokenizer_class(prompt_list, padding=False, truncation=True, add_special_tokens=True, max_length=max_length)
    target_tokens = tokenizer_class(output_list, padding=False, truncation=True, add_special_tokens=False, max_length=max_length) #Already has a EOS token when defining output_list

    input_ids = []
    attention_mask = []
    labels = []

    for i in range(len(init_list)):
        prompt_ids = prompt_tokens['input_ids'][i]
        target_ids = target_tokens['input_ids'][i]
        input_id = prompt_ids + target_ids
        attention_mask_list = [1] * len(input_id)
        label = [-100] * len(prompt_ids) + target_ids

        if len(input_id) > 2*max_length: # If combined length exceeds 2*max_length, truncate to 2*max_length (prompt + part of target)
            input_id = input_id[:2*max_length]
            attention_mask_list = attention_mask_list[:2*max_length]
            label = label[:2*max_length]  # Assumes prompt_length < 2*max_length. If prompt_length is too large, the sentence will be useless as a training example. Needs better handling.

        input_ids.append(input_id)
        attention_mask.append(attention_mask_list)
        labels.append(label)

    # Return as standard Python lists instead of numpy object arrays
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def tokenize_data(df: pd.DataFrame, model_name: str, cols: list, dir_path: str, max_length: int = 256, is_causal: bool = True):
    assert df is not None
    assert model_name is not None and isinstance(model_name, str)
    assert cols is not None and all(isinstance(col, str) for col in cols)

    dataset = Dataset.from_pandas(df)
    tokenizer = define_tokenizer(model_name, is_causal=is_causal)
    tokenized_data = dataset.map(tokenize_fn, fn_kwargs={'tokenizer_class': tokenizer, 'cols': cols, 'max_length': max_length}, batched=True, remove_columns=dataset.column_names)
    tokenized_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenizer.save_pretrained(dir_path)
    return tokenized_data

    