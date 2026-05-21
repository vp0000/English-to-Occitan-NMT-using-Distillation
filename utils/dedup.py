from typing import Optional
import pandas as pd
import regex as re
import hashlib

def dedup_code(
        data_frame: pd.DataFrame,
        src_col: str,
        tgt_col: str,
        score_col: Optional[str] = None,
        sort_flag: bool = True) -> pd.DataFrame:
    """
    Performs a two-pass deduplication on parallel corpus data.
    
    Pass 1: Remove duplicate pairs of (src_col, tgt_col) using hash-based comparison
    Pass 2: Remove duplicate values separately for src_col and tgt_col
    
    Args:
        data_frame: Input DataFrame with parallel sentences
        src_col: Source language column name (e.g., 'en')
        tgt_col: Target language column name (e.g., 'oc')
        score_col: Optional column name for quality scores. If provided and 
                   sort_before=True, sorts by this column (descending)
        sort_flag: If True, sorts by score_col before deduplication
    
    Returns:
        Deduplicated DataFrame
    """
    if sort_flag and score_col and score_col in data_frame.columns:
        data_frame = data_frame.sort_values(score_col, ascending=False)
    data_frame['hash'] = data_frame.apply(
        lambda x: hashlib.md5(
            f"{(re.sub(r'\s+', '', x[src_col]))}|{(re.sub(r'\s+', '', x[tgt_col]))}".encode()
        ).hexdigest(),
        axis=1
    )
    data_frame = data_frame.drop_duplicates('hash', keep='first')
    # Pass 2
    data_frame['src_base'] = data_frame[src_col].apply(lambda x: re.sub(r'\s+', '', str(x)))
    data_frame['tgt_base'] = data_frame[tgt_col].apply(lambda x: re.sub(r'\s+', '', str(x)))
    data_frame = data_frame.drop_duplicates('src_base', keep='first')
    data_frame = data_frame.drop_duplicates('tgt_base', keep='first')
    data_frame = data_frame.drop(columns=['hash', 'src_base', 'tgt_base'])

    return data_frame
