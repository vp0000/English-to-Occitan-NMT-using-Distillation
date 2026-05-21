from datasets import Dataset, load_dataset

def create_parallel_dataset(database_path: str, codes: list[str], cols: list[str], split: str = "dev") -> Dataset:
    flores_input = load_dataset(database_path, codes[0], split=split).to_pandas()
    flores_output = load_dataset(database_path, codes[1], split=split).to_pandas()
    return Dataset.from_dict({
        cols[0]: flores_input['text'],
        cols[1]: flores_output['text']
    })