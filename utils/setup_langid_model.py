def setup_langid_model(
    repo_url: str = "https://github.com/transducens/idiomata_cognitor.git",
    model_zip_path: str = "idiomata_cognitor/model.zip",
    extract_to: str = "model_files"
) -> str:
    """
    Clone the repo and extract the model if not already present.
    
    Returns:
        Path to the extracted .pkl model file
    """
    from pathlib import Path
    import subprocess
    import zipfile
    
    repo_path = Path("idiomata_cognitor")
    model_pkl_path = Path(extract_to) / "model.pkl"
    
    if model_pkl_path.exists():
        return str(model_pkl_path)
    
    # Clone repo if needed
    if not repo_path.exists():
        subprocess.run(["git", "clone", repo_url], check=True)
    
    # Extract model
    if Path(model_zip_path).exists():
        Path(extract_to).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    
    if not model_pkl_path.exists():
        raise FileNotFoundError(f"Could not find model at {model_pkl_path}")
    
    return str(model_pkl_path)