from utils import dedup, lang_identify
from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder
from tqdm.auto import tqdm
from pathlib import Path
from typing import Union, Optional
import regex as re
import unicodedata
import torch
import numpy as np
import pandas as pd

STATIC_JUNK_STARTS = set(
    " \t-–—.,;:?!|><^*#@~=+_\\/»«·•●★▲▼►◄←→↑↓©‹›\u00b7\ufeff"
)
VALID_ENDINGS = ('.', '?', '!', '"', '"', '»', "'", ')', ']')
VALID_STARTS_EXTRA = '"\'"«([¿¡'  # things that aren't alphanumeric but are valid sentence starts

def get_odd_punct(data: pd.DataFrame, src_col: str, tgt_col: str) -> set:
    def get_special_starts(series: list) -> set:
        # Extract the first character if it's not alphanumeric
        # Added the ord condition since '0' is not being caught by isalnum()
        return {s.strip()[0] for s in series if s and (not s.strip()[0].isalnum() or 48 <= ord(s[0]) < 58)}
    starts_src = get_special_starts([sent.strip() for sent in data[src_col].to_list()])
    starts_tgt = get_special_starts([sent.strip() for sent in data[tgt_col].to_list()])
    set_odd_punct = set()
    for char in (starts_src ^ starts_tgt):
        if char.isalnum() or char == '': continue
        set_odd_punct.add(char)
    return set_odd_punct

def precompile_regex_patterns(threshold: int, set_odd_punct: set[str]) -> dict:
    escaped = ''.join(map(re.escape, sorted(set_odd_punct, reverse=True)))
    return {
        'repeat': re.compile(r'(?P<punct>[!?,:;]){%d,}' % (threshold + 1)),
        'ellipsis': re.compile(r'(?P<ellipsis>[\.]){%d,}' % (threshold + 1)),
        'odd_punct': re.compile(f"^[{escaped}]+")
    }

def basic_cleaning(text: str, patterns: dict) -> str:
    """
    A basic cleaning function that uses regex patterns derived from heuristics and visual examination
    of sampled rows. Can be tinkered with if stricter cleaning is needed.
    """
    # Dictionaries for unicode characters to be replaced
    TYPO_MAP = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2013': '-', '\u2014': '-',
        '\u2026': '...', '\u00A0': ' '
    }

    SPACING_CHARS = {
        '\xa0': ' ', '\u200b': '', '\x9c': ''
    }

    # Normalizing unicode to avoid issues related to garbage characters
    text = unicodedata.normalize('NFC', text)

    # Initial removal of leading and trailing spaces
    text = text.strip()

    # 1. Basic Unicode cleaning
    # text = text.replace('\ufeff', '')
    text = re.sub(r'[\p{Cc}&&[^\n\t]]+', '', text)

    # 2. Normalize odd characters
    for dict_select in [TYPO_MAP, SPACING_CHARS]:
        for key, val in dict_select.items():
            text = text.replace(key, val)

    # 3. Remove whitespace before specific punctuation
    text = re.sub(r'\s+([!?.])', r'\1', text)
    # Keep only a single whitespace after specific punctuation to continue the sentence
    text = re.sub(r'([!?.])\s{2,}', r'\1 ', text)

    # 4. Collapse repeated punctuation
    text = patterns['repeat'].sub(lambda m: m.group('punct')*1, text)
    text = patterns['ellipsis'].sub(lambda m: m.group('ellipsis')*3, text) # Dots are handled separately to allow ellipsis
    
    # 5. Remove hanging opening and closing quotes (unbalanced if odd number of quotes)
    # Converting “ and ” to "" for now to avoid writing a whole BFS code
    # Leaving guillemets unchanged for now, this needs a BFS code to handle since we need guillemets in the Occitan sentences
    text = re.sub(r'“', r'"', text)
    text = re.sub(r'”', r'"', text)
    if sum(char == '"' for char in text) % 2 == 1:
        text = re.sub(r'"([^"]+$)', r'\1', text)
        text = re.sub(r'^([^"]*)"$', r'\1', text)
    if sum(char in ['«', '»'] for char in text) == 1:  # Removing single, unmatched guillemets for now, nested guillemets not handled yet
        text = re.sub(r'«([^«»]+$)', r'\1', text)
        text = re.sub(r'^([^«»]*)»$', r'\1', text)
    
    # 6. Remove unnecessary starting characters (anything except a digit, letter or quote mark, using heuristics for now)
    text = re.sub(r'^[»\s\-–—<<·↑→]+', r'', text)
    
    # 7. Remove unnecessary ending characters (anything except a punctuation mark, using heuristics to remove specific characters for now)
    text = re.sub(r'[=>>-–—«]+$', r'', text)
    
    # 8. Remove preceding spacing around colons, commas and other punctuation marks for phrases
    text = re.sub(r'\s+([:,;])', r'\1', text)
    # Keep only a single space around these marks
    text = re.sub(r'([:,;])\s{2,}', r'\1 ', text)
    
    # 9. Avoid spacing inside inverted commas, and keep only a single space around guillemets
    text = re.sub(r'"\s+([^""])', r'"\1', text)
    text = re.sub(r'([^""])\s+"', r'\1"', text)
    text = re.sub(r'«\s{2,}([^«»])', r'« \1', text)
    text = re.sub(r'([^«»])\s{2,}»', r'\1 »', text)
    
    # 10. Remove redundant punctuation from the start
    text = re.sub(patterns['odd_punct'], r'', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Final removal of leading and trailing spaces
    text = text.strip()

    return text

def is_complete_sentence(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if not text.endswith(VALID_ENDINGS):
        return False
    if not (text[0].isupper() or text[0].isdigit() or text[0] in VALID_STARTS_EXTRA):
        return False
    return True

def calculate_mults_and_filter(data: pd.DataFrame,
                               src_col: str,
                               tgt_col: str,
                               len_thres: int = 10,
                               word_thres: int = 3,
                               lower_mult_thres: int = 1,
                               upper_mult_thres: int = 3
                               ) -> pd.DataFrame:
    init_corpus = data[[src_col, tgt_col]].copy()
    init_corpus['word_cnt_src'] = init_corpus[src_col].str.strip().str.split().str.len()
    init_corpus['word_cnt_tgt'] = init_corpus[tgt_col].str.strip().str.split().str.len()
    init_corpus['len_src'] = init_corpus[src_col].str.strip().str.len()
    init_corpus['len_tgt'] = init_corpus[tgt_col].str.strip().str.len()
    # Will be using len_mult and word_cnt_mult as hardcoded columns since they are to be used in scoring
    init_corpus['len_mult'] = np.maximum(
        init_corpus['len_src'] / init_corpus['len_tgt'],
        init_corpus['len_tgt'] / init_corpus['len_src']
        )
    init_corpus['word_cnt_mult'] = np.maximum(
        init_corpus['word_cnt_src'] / init_corpus['word_cnt_tgt'],
        init_corpus['word_cnt_tgt'] / init_corpus['word_cnt_src']
        )
    init_corpus = init_corpus[
        (init_corpus['len_src'] >= len_thres) &
        (init_corpus['len_tgt'] >= len_thres) &
        (init_corpus['word_cnt_src'] >= word_thres) &
        (init_corpus['word_cnt_tgt'] >= word_thres)
    ]
    init_corpus = init_corpus[
        (init_corpus['len_mult'] >= lower_mult_thres) &
        (init_corpus['len_mult'] <= upper_mult_thres) &
        (init_corpus['word_cnt_mult'] >= lower_mult_thres) &
        (init_corpus['word_cnt_mult'] <= upper_mult_thres)
    ]
    # init_corpus = init_corpus.drop(columns=['len_src', 'len_tgt', 'word_cnt_src', 'word_cnt_tgt'])
    return init_corpus

# Labeling function
def language_labeling(
        data: pd.DataFrame,
        tgt_col: str,
        model_path: Union[str, Path]
) -> pd.DataFrame:
    identifier = lang_identify.load_identifier(model_path=model_path)
    data = identifier.add_labels_to_data(data, text_col=tgt_col)
    return data

    
def compute_similarity(
        data: pd.DataFrame,
        src_col: str,
        tgt_col: str,
        label_col: str,
        lang_map: dict[str, str],
        tokenizer,
        encoder,
        chunk_size: int = 10000
) -> pd.DataFrame:
    
    def get_embeddings(tokenizer, encoder, text_list, lang="eng_Latn", batch_size=128):
    # Helper function to tokenize text using FLORES-200 language code conventions
        tokenizer.src_lang = lang
        all_embeddings = []

        for i in range(0, len(text_list), batch_size):
            batch_text = text_list[i:i+batch_size]
            inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # Move inputs to the same device as the encoder
            inputs = {k: v.to(encoder.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = encoder(**inputs)
            # Pool embeddings over the sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Return as numpy array on CPU for compatibility with downstream numpy operations
            all_embeddings.append(embeddings.cpu().numpy())
            # Free up memory
            del inputs, outputs
            torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)
    
    data['lang_code'] = data[label_col].map(lang_map)
    concat_list = []
    for lang_code in set(data['lang_code']):
        matching_languages = [k for k, v in lang_map.items() if v == lang_code]
        print(f"Processing Languages: {' ,'.join(matching_languages)}")
        extracted_data = data[data['lang_code'] == lang_code].copy()
        curr_chunk_size = chunk_size
        if len(extracted_data) == 0: continue
        if len(extracted_data) < curr_chunk_size: curr_chunk_size = len(extracted_data)
        all_similarities = np.zeros(len(extracted_data))
        for start_idx in tqdm(range(0, len(extracted_data), curr_chunk_size), desc="Processing chunks"):
            end_idx = start_idx + curr_chunk_size
            # end_idx = min(end_idx, len(extracted_data))
            chunk = extracted_data.iloc[start_idx:end_idx].copy()
            en_embeddings = get_embeddings(tokenizer, encoder, chunk[src_col].tolist())
            tgt_embeddings = get_embeddings(tokenizer, encoder, chunk[tgt_col].tolist(), lang=lang_code)
            similarities = np.sum(en_embeddings * tgt_embeddings, axis=1) \
                  / (np.linalg.norm(en_embeddings, axis=1) * np.linalg.norm(tgt_embeddings, axis=1))
            all_similarities[start_idx:end_idx] = similarities
        extracted_data['sonar_sim'] = all_similarities
        concat_list.append(extracted_data)

    parallel_data = pd.concat(concat_list)
    parallel_data = parallel_data.reindex(data.index)
    return parallel_data

def final_filtering(
        data: pd.DataFrame,
        confidence_dict: dict,
        src_col: str,
        tgt_col: str,
        quantile_thres: float = 0.1
        ) -> pd.DataFrame:
    assert 'sonar_sim' in data.columns
    assert src_col and src_col in data.columns
    assert tgt_col and tgt_col in data.columns
    data['sim_score'] = (data['lang_code'].map(confidence_dict))*data['sonar_sim']
    word_cmpt = np.exp(-(data['word_cnt_mult'] - 1))
    len_cmpt = np.exp(-(data['len_mult'] - 1))
    len_penalty = np.clip(np.minimum(data['word_cnt_src'], data['word_cnt_tgt']) / 8, 0, 1)
    data['final_score'] = 0.75*data['sim_score'] + 0.15*word_cmpt + 0.1*len_cmpt
    data['final_score'] = data['final_score']*(len_penalty)
    data = data[data['final_score'] >= np.quantile(data['final_score'], quantile_thres)]
    data = pd.DataFrame(data[[src_col, tgt_col, 'final_score']]) # Ensuring final result is a dataframe
    return data

def cleaning_orchestrator(
        df: pd.DataFrame,
        lang_map: dict,
        confidence_dict: dict,
        src_col: str = 'en',
        tgt_col: str = 'oc',
        punct_threshold: int = 3,
        len_thres: int = 10,
        word_thres: int = 3,
        lower_mult_thres: int = 1,
        upper_mult_thres: int = 3,
        label_col: str = 'label',
        lang_id_model_path: Union[str, Path] = "model_files/model.pkl",
        embed_model_name: str = "cointegrated/SONAR_200_text_encoder",
        chunk_size: int = 10000,
        quantile_thres: float = 0.1,
        output_path: Optional[str] = None
) -> pd.DataFrame:
    data = df.copy()
    data = dedup.dedup_code(data, src_col, tgt_col, score_col='scores', sort_flag=True)
    set_odd_punct = get_odd_punct(data, src_col, tgt_col)
    set_odd_punct = set_odd_punct | STATIC_JUNK_STARTS
    precompiled_patterns = precompile_regex_patterns(punct_threshold, set_odd_punct)
    data[src_col] = data[src_col].apply(lambda x: basic_cleaning(x, precompiled_patterns))
    data[tgt_col] = data[tgt_col].apply(lambda x: basic_cleaning(x, precompiled_patterns))
    data = dedup.dedup_code(data, src_col, tgt_col) # No need for sorting by the old scores
    data['src_complete'] = data[src_col].apply(is_complete_sentence)
    data['tgt_complete'] = data[tgt_col].apply(is_complete_sentence)
    data = pd.DataFrame(data[data['src_complete'] & data['tgt_complete']])
    data = data.drop(columns=['src_complete', 'tgt_complete'])
    data = calculate_mults_and_filter(
        data,
        src_col=src_col,
        tgt_col=tgt_col,
        len_thres=len_thres,
        word_thres=word_thres,
        lower_mult_thres=lower_mult_thres,
        upper_mult_thres=upper_mult_thres
        )
    data = language_labeling(data, tgt_col=tgt_col, model_path=lang_id_model_path)
    tokenizer = AutoTokenizer.from_pretrained(embed_model_name, device_map="auto")
    encoder = M2M100Encoder.from_pretrained(embed_model_name, device_map="auto")
    data = compute_similarity(
        data,
        src_col=src_col,
        tgt_col=tgt_col,
        label_col=label_col,
        lang_map=lang_map,
        tokenizer=tokenizer,
        encoder=encoder,
        chunk_size=chunk_size
    )
    data = final_filtering(
        data,
        confidence_dict=confidence_dict,
        src_col=src_col,
        tgt_col=tgt_col,
        quantile_thres=quantile_thres
    )
    assert output_path is not None
    if 'xlsx' in output_path:
        data.to_excel(output_path, index=False)
    elif 'csv' in output_path:
        data.to_csv(output_path, index=False, encoding="utf-8-sig")
    return data
    
        