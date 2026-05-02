import os
from typing import Optional

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