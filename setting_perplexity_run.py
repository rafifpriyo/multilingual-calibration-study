import lm_eval
import os
import shutil
util_source_path = f"./edited_vllm_causallms.py"
util_update_path = f"{os.path.join(os.path.dirname(lm_eval.__file__), f'models/vllm_causallms.py')}"

shutil.copyfile(util_source_path, util_update_path)