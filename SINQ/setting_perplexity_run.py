import lm_eval
import evaluate
import os
import shutil
# LM Eval Harness edit
# Max length error, extended 1 token
util_source_path = f"../edited_vllm_causallms.py"
util_update_path = f"{os.path.join(os.path.dirname(lm_eval.__file__), f'models/vllm_causallms.py')}"

shutil.copyfile(util_source_path, util_update_path)

# Multiblimp datasets
shutil.copytree("../BHASA", "./BHASA")
shutil.copytree("../syntactic_generalization_multilingual", "./syntactic_generalization_multilingual")
shutil.copytree("../zhoblimp", "./zhoblimp")

# True False answers for flips metric
util_source_path = f"../edited_lm_eval/metrics.py"
util_update_path = f"{os.path.join(os.path.dirname(lm_eval.__file__), f'api/metrics.py')}"

shutil.copyfile(util_source_path, util_update_path)

util_source_path = f"../edited_lm_eval/task.py"
util_update_path = f"{os.path.join(os.path.dirname(lm_eval.__file__), f'api/task.py')}"

shutil.copyfile(util_source_path, util_update_path)

util_source_path = f"../edited_lm_eval/evaluator.py"
util_update_path = f"{os.path.join(os.path.dirname(lm_eval.__file__), f'evaluator.py')}"

shutil.copyfile(util_source_path, util_update_path)

util_source_path = f"../edited_lm_eval/evaluator_utils.py"
util_update_path = f"{os.path.join(os.path.dirname(lm_eval.__file__), f'evaluator_utils.py')}"

shutil.copyfile(util_source_path, util_update_path)

# Huggingface Evaluate's Perplexity
util_source_path = f"./perplexity.py"
util_update_path = f"{os.path.join(os.path.dirname(evaluate.__file__), f'metrics/perplexity/perplexity.py')}"

import os
if not os.path.exists(os.path.dirname(util_update_path)):
    os.makedirs(os.path.dirname(util_update_path))

shutil.copyfile(util_source_path, util_update_path)