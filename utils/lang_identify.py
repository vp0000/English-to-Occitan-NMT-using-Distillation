import joblib
from pathlib import Path
from typing import List, Union
import pandas as pd

LANGUAGE_LABELS = { 
    1.: 'Spanish',
    2.: 'Catalan',
    3.: 'Aragonese',
    4.: 'Aranese',
    5.: 'Occitan',
    6.: 'Asturian',
    7.: 'Galician',
    8.: 'Italian',
    9.: 'French',
    10.: 'Portuguese'
}

class Language_Identifier: # Class-based structure to allow the idiomata_cognitor files to be used without argparse
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self._model = None
    @property
    def model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            self._model = joblib.load(self.model_path)
        return self._model
    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predicts labels for a set of texts or a single text, returning a list of labels or a single label as applicable.
        """
        assert isinstance(texts, str) or (isinstance(texts, List) and all(isinstance(txt, str) for txt in texts))
        single_input = isinstance(texts, str)
        if single_input: texts = [texts]
        predictions = self.model.predict(texts)
        labels = [LANGUAGE_LABELS[pred] for pred in predictions]
        return labels[0] if single_input else labels
    def predict_proba(self, texts: Union[str, List[str]]) -> Union[dict, List[dict]]:
        """
        Returns predicted probabilities for each of the possible classes in LANGUAGE_LABELS.
        The output is a single dictionary of probabilities or a list of such dictionaries depending on the input.
        """
        assert isinstance(texts, str) or (isinstance(texts, List) and all(isinstance(txt, str) for txt in texts))
        if isinstance(texts, str): texts = [texts]
        probabilities = self.model.predict_proba(texts)
        results = []
        for probs in probabilities:
            result = {}
            for i, prob in enumerate(probs):
                if prob > 0:
                    label = LANGUAGE_LABELS.get(self.model.classes_[i], f"Unknown_class_{i}")
                    result[label] = prob
            results.append(result)
        return results[0] if len(results) == 1 else results
    def add_labels_to_data(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Adds labels to a dataframe by using the predict function on a given column.
        """
        df = df.copy()
        df['label'] = self.predict(df[text_col].to_list())
        return df

def load_identifier(model_path: Union[str, Path]) -> Language_Identifier:
    """
    Helper function to initialize the Language_Identifier object.
    """
    return Language_Identifier(model_path=model_path)

def predict_language(texts: Union[str, List[str]], model_path: Union[str, Path]) -> Union[str, List[str]]:
    """
    Predicts labels for a set of texts or a single text using the Language_Identifier class.
    """
    identifier = load_identifier(model_path=model_path)
    return identifier.predict(texts=texts)

    