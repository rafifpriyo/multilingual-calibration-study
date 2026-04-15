import lm_eval
import evaluate
import os
import shutil
util_source_path = f"./edited_vllm_causallms.py"
util_update_path = f"{os.path.join(os.path.dirname(lm_eval.__file__), f'models/vllm_causallms.py')}"

shutil.copyfile(util_source_path, util_update_path)

util_source_path = f"./perplexity.py"
util_update_path = f"{os.path.join(os.path.dirname(evaluate.__file__), f'metrics/perplexity/perplexity.py')}"

import os
if not os.path.exists(os.path.dirname(util_update_path)):
    os.makedirs(os.path.dirname(util_update_path))

shutil.copyfile(util_source_path, util_update_path)