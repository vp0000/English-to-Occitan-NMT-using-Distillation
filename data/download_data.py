import requests
import os
import zipfile
import pandas as pd
from typing import Optional
from tqdm import tqdm

def download_corpus(url: str, corpus: str, source: str, target: str, dir_path: str,
                  preprocessing: str, chunk_size: int = 256):
    opus_url_curr = (url + f"corpus={corpus}&source={source}&" +
                     f"target={target}&preprocessing={preprocessing}")
    dataset_info = requests.get(url=opus_url_curr).json()
    if not dataset_info['corpora']:
        print(f"No corpus found for {corpus}, {source}, {target}, {preprocessing}")
        return None

    dataset_url = dataset_info['corpora'][0]['url']
    # Extract filename from dataset_url to preserve original extension (e.g., .tmx.gz)
    file_name = os.path.basename(dataset_url)
    file_path = os.path.join(dir_path, corpus)
    os.makedirs(file_path, exist_ok=True) # Create directory if it doesn't exist
    dest_path = os.path.join(file_path, file_name)

    with requests.get(url=dataset_url, stream=True) as response:
        try:
            response.raise_for_status()
            with open(dest_path, 'wb') as file:
                for chunk in tqdm(response.iter_content(chunk_size=chunk_size), desc=f"Downloading {file_name}"):
                    file.write(chunk)
            print(f"Downloaded {file_name}")
            return dest_path # Return the filepath for later use
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error during download: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
            return None
    return None