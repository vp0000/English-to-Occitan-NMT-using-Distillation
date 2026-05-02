from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from sacrebleu import corpus_bleu, corpus_chrf

def generate_translations_for_eval(
        model: AutoModelForCausalLM|AutoModelForSeq2SeqLM,
        texts: dict,
        tokenizer: AutoTokenizer,
        batch_size: int = 8,
        model_type: str = "causal",
        lang_code: str = "oci") -> list[dict]:
    processed_outputs = []
    total = len(texts['en'])
    model.eval().to("cuda")

    with torch.inference_mode():
        for i in tqdm(range(0, total, batch_size), desc="Generating"):
            en_batch = texts['en'][i:i + batch_size]
            oc_gold_batch = texts['oc'][i:i + batch_size]

            if model_type == "causal":
                prompts = [f"Translate English to Occitan: {en} -> " for en in en_batch]

                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                tokenizer.padding_side = "left"

                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to("cuda")

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    num_beams=5,
                    max_length=None, # Explicitly set to None to avoid conflict with max_new_tokens
                    # early_stopping=True,
                    # length_penalty=1.0,
                    eos_token_id=tokenizer.eos_token_id
                    # no_repeat_ngram_size=4,
                    # repetition_penalty=1.5
                )

                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for src, gold_oc, gen_text in zip(en_batch, oc_gold_batch, decoded):
                    # Strip prompt
                    prompt = f"Translate English to Occitan: {src} -> "
                    trans = gen_text.replace(prompt, '', 1).split("\n")[0].strip()
                    trans = trans.rstrip(" .,;:")

                    processed_outputs.append({
                    "en": src,
                    "oc_gold": gold_oc,
                    "oc_synthetic": trans
                    })

            elif model_type == "seq2seq":
                input_batch = [f">>{lang_code}<< {src}" for src in en_batch]

                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                tokenizer.padding_side = "right"

                inputs =  tokenizer(
                    input_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to("cuda")

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    num_beams=5,
                    max_length=None, # Explicitly set to None to avoid conflict with max_new_tokens
                    eos_token_id=tokenizer.eos_token_id
                )

                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for src, gold_oc, trans in zip(en_batch, oc_gold_batch, decoded):
                    # Strip prompt
                    # prompt = f"Translate English to Occitan: {src} -> "
                    # trans = gen_text.replace(prompt, '', 1).strip()
                    # trans = trans.rstrip(" .,:")
                    trans = trans.strip()

                    processed_outputs.append({
                    "en": src,
                    "oc_gold": gold_oc,
                    "oc_synthetic": trans
                    })

            if i % 20 == 0:
                torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return processed_outputs

def score_bleu_chrf(trans_dict: list[dict]) -> dict:
    hyps = [t["oc_synthetic"] for t in trans_dict]
    refs = [t["oc_gold"] for t in trans_dict]
    assert len(hyps) == len(refs), f"len mismatch: {len(hyps)} vs {len(refs)}"
    bleu = corpus_bleu(hyps, [refs])
    chrf = corpus_chrf(hyps, [refs])
    # print(f"[{label}] BLEU: {bleu.score:.2f} | chrF: {chrf.score:.2f}")
    # print("sacreBLEU signature:", bleu) # Modified to print the object directly
    return {"bleu": bleu.score, "chrf": chrf.score, "sig": bleu}

def generate_eval_metrics(
        model,
        texts,
        tokenizer,
        batch_size=8,
        model_type="causal",
        lang_code="oci",
    ):
    translations = generate_translations_for_eval(model=model, texts=texts, tokenizer=tokenizer, batch_size=batch_size, model_type=model_type, lang_code=lang_code)
    model_metrics = score_bleu_chrf(translations)
    return model_metrics