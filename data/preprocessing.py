import pandas as pd
from typing import Optional
import os
import zipfile

def create_corpus(file_path: Optional[str], corpus_name: str) -> Optional[pd.DataFrame]:
    assert file_path is not None
    file_name = os.path.basename(file_path)
    dir_path = os.path.dirname(file_path)
    file_id = file_name.split('.')[0]
    exts = file_id.split('-')
    exts.append('scores')
    data_frame = pd.DataFrame()
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dir_path)
        file_names = [f"{corpus_name}.{file_id}.{ext}" for ext in exts]
        for i, file_name in enumerate(file_names):
            list_lines = []
            file_path = os.path.join(dir_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        list_lines.append(line.strip())
                if exts[i] not in data_frame.columns:
                    data_frame[exts[i]] = list_lines
            except FileNotFoundError:
                print(f"Error: The file {file_path} was not found.")
                return None
    return data_frame


def drop_duplicate_rows(data_frame: Optional[pd.DataFrame], cols: Optional[list[str]]) -> pd.DataFrame:
    #This function drops duplicate rows based on the assumption that the similarity scoring is descending.
    assert data_frame is not None
    assert isinstance(data_frame, pd.DataFrame)
    assert cols is not None and all(isinstance(col, str) for col in cols)
    if cols is not None:
        print(f"Initial row count: {len(data_frame)}")
        df_copy = data_frame.copy()
        for col in cols:
            if col not in df_copy.columns:
                print(f"Warning: Column '{col}' not found in DataFrame. Skipping this column.")
                continue
            df_copy[col] = df_copy[col].str.strip()
            df_copy[f'{col}_regex'] = df_copy[col].str.replace(r'\s+', '', regex=True)
        df_trunc = df_copy.drop_duplicates(subset=[f'{col}_regex' for col in cols], keep='first')
        print(f"New row count after dropping duplicates based on columns {cols}: {len(df_trunc)}")
        for col in cols:
            df_trunc.drop_duplicates(subset=[f'{col}_regex'], keep='first', inplace=True)
            print(f"New row count after dropping duplicates based on '{col}': {len(df_trunc)}")
        return df_trunc.drop(columns=[f'{col}_regex' for col in cols])
    else:
        df_trunc = data_frame.drop_duplicates(keep='first')
        print(f"New row count after dropping duplicates based on all columns: {len(df_trunc)}")
        return df_trunc
    
def preprocess_corpus(file_path: Optional[str], corpus_name: str, cols: Optional[list[str]]) -> Optional[pd.DataFrame]:
    assert file_path is not None
    assert corpus_name is not None
    assert cols is not None and all(isinstance(col, str) for col in cols)
    data_frame = create_corpus(file_path, corpus_name)
    if data_frame is not None:
        return drop_duplicate_rows(data_frame, cols)
    else:
        print("Error: Failed to create corpus. Preprocessing aborted.")
        return None