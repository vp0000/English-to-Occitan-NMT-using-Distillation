from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from sacrebleu import corpus_bleu, corpus_chrf
import pandas as pd

def _generate_causal(model, texts, tokenizer, batch_size, num_beams, max_new_tokens):
    """Generate translations using an Unsloth-loaded causal LM."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    en_list = list(texts["en"])
    oc_list = list(texts["oc"])

    # Sort by length to minimize padding waste; remember original order
    sorted_idx = sorted(range(len(en_list)), key=lambda i: len(en_list[i]))
    en_sorted = [en_list[i] for i in sorted_idx]
    oc_sorted = [oc_list[i] for i in sorted_idx]

    sorted_outputs = []
    for i in tqdm(range(0, len(en_sorted), batch_size), desc="Generating (causal)"):
        en_batch = en_sorted[i : i + batch_size]
        oc_batch = oc_sorted[i : i + batch_size]
        prompts = [f"Translate the following English sentence to Occitan: {s} ->" for s in en_batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                # cache_implementation="dynamic"
            )

        gen_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for src, gold, trans in zip(en_batch, oc_batch, decoded):
            sorted_outputs.append({
                "en": src,
                "oc_gold": gold,
                "oc_synthetic": trans.strip(),
            })

    # Restore input order
    reordering = [0] * len(sorted_idx)
    for original, sorted_pos in enumerate(sorted_idx):
        reordering[sorted_pos] = original
    return [sorted_outputs[reordering[i]] for i in range(len(sorted_outputs))]

def _generate_seq2seq(model, texts, tokenizer, batch_size, num_beams, max_new_tokens, lang_code):
    """Generate translations using a standard seq2seq model (Marian baseline)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    en_list = list(texts["en"])
    oc_list = list(texts["oc"])

    sorted_idx = sorted(range(len(en_list)), key=lambda i: len(en_list[i]))
    en_sorted = [en_list[i] for i in sorted_idx]
    oc_sorted = [oc_list[i] for i in sorted_idx]

    sorted_outputs = []
    for i in tqdm(range(0, len(en_sorted), batch_size), desc="Generating (seq2seq)"):
        en_batch = en_sorted[i : i + batch_size]
        oc_batch = oc_sorted[i : i + batch_size]
        input_batch = [f">>{lang_code}<< {s}" for s in en_batch]

        inputs = tokenizer(
            input_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for src, gold, trans in zip(en_batch, oc_batch, decoded):
            sorted_outputs.append({
                "en": src,
                "oc_gold": gold,
                "oc_synthetic": trans.strip(),
            })

    reordering = [0] * len(sorted_idx)
    for original, sorted_pos in enumerate(sorted_idx):
        reordering[sorted_pos] = original
    return [sorted_outputs[reordering[i]] for i in range(len(sorted_outputs))]

def _generate_nllb(model, texts, tokenizer, batch_size, num_beams, max_new_tokens,
                  src_lang="eng_Latn", tgt_lang="oci_Latn"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print("oci_Latn token id:", tokenizer.convert_tokens_to_ids("oci_Latn"))
    print("unk token id:     ", tokenizer.unk_token_id)
    print("eos token id:     ", tokenizer.eos_token_id)
    # print("Special tokens:   ", tokenizer.add_special_tokens[:20])

    en_list = list(texts["en"])
    oc_list = list(texts["oc"])

    sorted_idx = sorted(range(len(en_list)), key=lambda i: len(en_list[i]))
    en_sorted = [en_list[i] for i in sorted_idx]
    oc_sorted = [oc_list[i] for i in sorted_idx]

    sorted_outputs = []
    for i in tqdm(range(0, len(en_sorted), batch_size), desc="Generating (seq2seq)"):
        en_batch = en_sorted[i : i + batch_size]
        oc_batch = oc_sorted[i : i + batch_size]
    
        inputs = tokenizer(en_batch, return_tensors="pt", padding=True,
                       truncation=True, max_length=256).to(device)
    
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True if num_beams > 1 else False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                length_penalty=0.8,
                use_cache=True,
            )
    
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for src, gold, trans in zip(en_batch, oc_batch, decoded):
            sorted_outputs.append({
                "en": src,
                "oc_gold": gold,
                "oc_synthetic": trans.strip(),
            })

    reordering = [0] * len(sorted_idx)
    for original, sorted_pos in enumerate(sorted_idx):
        reordering[sorted_pos] = original
    return [sorted_outputs[reordering[i]] for i in range(len(sorted_outputs))]

def generate_translations_for_eval(
    model,
    texts: dict,
    tokenizer,
    batch_size: int = 8,
    model_type: str = "causal",
    lang_code: str = "oci",
    src_lang: str = "eng_Latn",
    tgt_lang: str = "oci_Latn",
    num_beams: int = 5,
    max_new_tokens: int = 128,
    nllb_flag: bool = True
) -> list[dict]:
    if model_type == "causal":
        return _generate_causal(model, texts, tokenizer, batch_size, num_beams, max_new_tokens)
    elif model_type == "seq2seq":
        if nllb_flag: return _generate_nllb(model, texts, tokenizer, batch_size, num_beams, max_new_tokens, src_lang, tgt_lang)
        else: return _generate_seq2seq(model, texts, tokenizer, batch_size, num_beams, max_new_tokens, lang_code)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")


def score_bleu_chrf(trans_dict: list[dict]) -> dict:
    hyps = [t["oc_synthetic"] for t in trans_dict]
    refs = [t["oc_gold"] for t in trans_dict]
    assert len(hyps) == len(refs), f"len mismatch: {len(hyps)} vs {len(refs)}"
    bleu = corpus_bleu(hyps, [refs])
    chrf = corpus_chrf(hyps, [refs])
    return {"bleu": bleu.score, "chrf": chrf.score, "sig": bleu}


def generate_eval_metrics(
    model,
    texts,
    tokenizer,
    output_path: str,
    batch_size: int = 8,
    model_type: str = "causal",
    lang_code: str = "oci",
    src_lang: str = "eng_Latn",
    tgt_lang: str = "oci_Latn",
    num_beams: int = 1,
    max_new_tokens: int = 128,
    nllb_flag: bool = True
):
    translations = generate_translations_for_eval(
        model=model,
        texts=texts,
        tokenizer=tokenizer,
        batch_size=batch_size,
        model_type=model_type,
        lang_code=lang_code,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        nllb_flag=nllb_flag
    )
    data = pd.DataFrame(translations)
    assert output_path is not None
    if 'xlsx' in output_path:
        data.to_excel(output_path, index=False)
    elif 'csv' in output_path:
        data.to_csv(output_path, index=False)
    return score_bleu_chrf(translations)