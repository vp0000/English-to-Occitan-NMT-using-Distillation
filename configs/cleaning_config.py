from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
import os

@dataclass
class Cleaning_Config:
    """Default configuration for cleaning the parallel corpus."""
    lang_map: dict = field(default_factory = lambda: {
    "Occitan": "oci_Latn",
    "Aranese": "oci_Latn",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "Catalan": "cat_Latn",
    "Aragonese": "cat_Latn",
    "Asturian": "spa_Latn",
    "Galician": "spa_Latn",
    "Portuguese": "por_Latn",
    "Italian": "ita_Latn"
})
    confidence_dict: dict = field(default_factory = lambda: {
    "oci_Latn": 1,
    "cat_Latn": 0.9,
    "fra_Latn": 0.7,
    "spa_Latn": 0.6,
    "por_Latn": 0.5,
    "ita_Latn": 0.4
}) # Heuristic confidence multiple to weight sentence similarity
    
    punct_threshold: int = 3
    len_thres: int = 10
    word_thres: int = 3
    lower_mult_thres: int = 1
    upper_mult_thres: int = 3
    label_col: str = 'label'
    # lang_id_model_path: Union[str, Path] = "model_files/model.pkl"
    embed_model_name: str = "cointegrated/SONAR_200_text_encoder"
    chunk_size: int = 10000
    quantile_thres: float = 0.1
    # output_path: str = field(default_factory=lambda: os.path.join(os.getcwd(), "datasets/cleaned_data.csv"))