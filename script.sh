#!/bin/bash

languages=("English" "Indonesian" "Tamil" "Swahili" "Chinese")
bits=(8 4 3 2)
quantization_techniques=("tacq" "slimllm" "gptq")

for quant in "${quantization_techniques[@]}"; do
  for bit in "${bits[@]}" ; do
    for lang in "${languages[@]}" ; do
        pixi run python -m multilingual_evaluation_floresplus_perplexity --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-floresplus-qwen-'$quant$lang$bit
        pixi run python -m multilingual_evaluation_include_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-include-qwen-'$quant$lang$bit
        pixi run python -m multilingual_evaluation_globalmmlulite_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-globalmmlulite-qwen-'$quant$lang$bit
        pixi run python -m multilingual_evaluation_belebele_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-belebele-qwen-'$quant$lang$bit
        pixi run python -m multilingual_evaluation_multiblimp_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-multiblimp-qwen-'$quant$lang$bit
        pixi run python -m multilingual_evaluation_wikipedia_perplexity --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-wikipedia-qwen-'$quant$lang$bit

        rm -rf '/workspace/.hf_home/hub/models--fifrio--Qwen3-8B-'$quant'-'$bit'bit-calibration-'$lang'-128samples'
    done
  done
done

rm -rf '/workspace/.hf_home/hub/models--Qwen--Qwen3-8B'