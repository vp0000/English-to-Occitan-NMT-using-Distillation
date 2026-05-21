import os
from typing import Optional
from pathlib import Path

def check_checkpoint(checkpt: Optional[str|bool], def_path: str) -> Optional[str|bool]:
    assert isinstance(def_path, str)
    assert isinstance(checkpt, str) or isinstance(checkpt, bool) or checkpt is None
    if checkpt == None or checkpt == False: return None
    elif checkpt == True:
        try:
            if any('checkpoint' in fld for fld in os.listdir(def_path)): return True
            else: return None
        except OSError as e:
            print(f'Error: {e}')
    else:
        if os.path.isdir(checkpt): return checkpt
        else: return None

def model_exists(model_dir: str) -> bool:
    """Check whether a complete, loadable LoRA/PEFT model exists at model_dir."""
    path = Path(model_dir)
    if not path.is_dir():
        return False
    
    # PEFT adapter config (required to know which base model to load)
    if not (path / "adapter_config.json").is_file():
        return False
    
    # Adapter weights — newer saves use safetensors, older use .bin
    weights_files = ["adapter_model.safetensors", "adapter_model.bin"]
    has_weights = any(
        (path / w).is_file() and (path / w).stat().st_size > 0
        for w in weights_files
    )
    if not has_weights:
        return False
    
    # training_config.json is written last by save_model_config, so its presence
    # confirms the full save process completed (and not just a partial save)
    if not (path / "training_config.json").is_file():
        return False
    
    return True
