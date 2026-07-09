# Argument
import argparse
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig

from dotenv import load_dotenv
from huggingface_hub import create_repo
from huggingface_hub import HfApi

def quantize_sinq(model_name, device, bit, save_dir, group_size=128, random_seed=1234):
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    
    torch_dtype = "bfloat16" if "aya" not in model_name else "float16"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True if "aya" in model_name else False)

    quant_cfg = BaseQuantizeConfig(
        nbits=bit,            # quantization bit-width
        group_size=group_size,      # group size
        tiling_mode="1D",   # tiling strategy
        method="sinq"       # quantization method ("asinq" for the calibrated version)
    )

    qmodel = AutoSINQHFModel.quantize_model(
        model,
        tokenizer=tokenizer,
        quant_config=quant_cfg,
        compute_dtype=torch_dtype,
        device=device
    )

    if save_dir:
        # 'model' must already be SINQ-quantized (e.g., via AutoSINQHFModel.quantize_model)
        AutoSINQHFModel.save_quantized_safetensors(
            qmodel,
            tokenizer,
            save_dir,
            verbose=True,
            max_shard_size="4GB",   # typical HF shard size (use "8GB" if you prefer)
        )

    return qmodel

if __name__ == "__main__":
    """# Parameter"""
    load_dotenv()

    if "HF_KEY" not in os.environ:
        raise EnvironmentError("HF_KEY environment variable is not defined. Please set it before running the application.")
    if "WANDB_KEY" not in os.environ:
        raise EnvironmentError("WANDB_KEY environment variable is not defined. Please set it before running the application.")

    hf_key = os.environ["HF_KEY"]
    wandb_key = os.environ["WANDB_KEY"]

    parser = argparse.ArgumentParser("args_gptq")
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--bit", type=int)
    parser.add_argument("--random_seed", type=int, default=1234)

    args = parser.parse_args()
    model_id = args.model_id
    bit = args.bit
    random_seed = args.random_seed
    
    # Specify save dir with its name
    save_dir = f"./output/{model_id}-{bit}bit" + "" if random_seed == 1234 else f"-{random_seed}randomseed"

    qmodel = quantize_sinq(model_id, "cuda", bit, save_dir random_seed=random_seed)

    """# Upload to Huggingface"""
    repo_name = f"fifrio/{model_id.split('/')[-1]}-sinq-{bit}bit-128samples{('-' + str(random_seed) + 'randomseed') if random_seed != 1234 else ''}"

    create_repo(repo_name, repo_type="model", token=hf_key)

    api = HfApi(token=hf_key)

    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_name,
        repo_type="model",
    )