#!/bin/bash

# pixi run python -m multilingual_evaluation_belebele-chinese_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique Unquantized --bit 32 2>&1 | tee 'log-belebele-qwen-Unquantized'
# pixi run python -m multilingual_evaluation_mmluproxlite_generateuntil --model_id Qwen/Qwen3-8B --quantization_technique Unquantized --lang Unquantized --bit 32 2>&1 | tee 'log-mmluproxlite-qwen-Unquantized'

bits=(4 3 2)
quantization_techniques=("sinq")

for quant in "${quantization_techniques[@]}"; do
  for bit in "${bits[@]}" ; do
      pixi run python -m multilingual_evaluation_floresplus_perplexity --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --bit "$bit" --nsamples 128 2>&1 | tee 'log-floresplus-qwen-'$quant$bit
      pixi run python -m multilingual_evaluation_include_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --bit "$bit" --nsamples 128 2>&1 | tee 'log-include-qwen-'$quant$bit
      # pixi run python -m multilingual_evaluation_mmluproxlite_generateuntil --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --bit "$bit" --nsamples 128 2>&1 | tee 'log-mmluproxlite-qwen-'$quant$bit
      pixi run python -m multilingual_evaluation_globalmmlulite_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --bit "$bit" --nsamples 128 2>&1 | tee 'log-globalmmlulite-qwen-'$quant$bit
      pixi run python -m multilingual_evaluation_belebele_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --bit "$bit" --nsamples 128 2>&1 | tee 'log-belebele-qwen-'$quant$bit
      pixi run python -m multilingual_evaluation_multiblimp_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --bit "$bit" --nsamples 128 2>&1 | tee 'log-multiblimp-qwen-'$quant$bit
      pixi run python -m multilingual_evaluation_massive-scenario_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --bit "$bit" --nsamples 128 2>&1 | tee 'log-multiblimp-qwen-'$quant$bit
      pixi run python -m multilingual_evaluation_xstorycloze_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --bit "$bit" --nsamples 128 2>&1 | tee 'log-multiblimp-qwen-'$quant$bit
      # pixi run python -m multilingual_evaluation_wikipedia_perplexity --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --bit "$bit" --nsamples 128 2>&1 | tee 'log-wikipedia-qwen-'$quant$bit

      rm -rf '/workspace/.hf_home/hub/models--fifrio--Qwen3-8B-'$quant'-'$bit'bit-128samples'
  done
done

rm -rf '/workspace/.hf_home/hub/models--Qwen--Qwen3-8B'