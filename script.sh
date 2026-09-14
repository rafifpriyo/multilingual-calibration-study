#!/bin/bash

pixi run python -m multilingual_evaluation_belebele_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique Unquantized --lang Unquantized --bit 32 2>&1 | tee 'log-belebele-qwen-Unquantized'
# pixi run python -m multilingual_evaluation_mmluproxlite_generateuntil --model_id Qwen/Qwen3-8B --quantization_technique Unquantized --lang Unquantized --bit 32 2>&1 | tee 'log-mmluproxlite-qwen-Unquantized'

languages=("English" "Indonesian" "Tamil" "Swahili" "Chinese")
bits=(4 3 2)
quantization_techniques=("tacq" "slimllm" "gptq")

for quant in "${quantization_techniques[@]}"; do
  for bit in "${bits[@]}" ; do
    for lang in "${languages[@]}" ; do
        # pixi run python -m multilingual_evaluation_include_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-include-qwen-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_floresplus_perplexity --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-floresplus-qwen-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_mmluproxlite_generateuntil --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-mmluproxlite-qwen-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_globalmmlulite_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-globalmmlulite-qwen-'$quant$lang$bit
        pixi run python -m multilingual_evaluation_belebele_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-belebele-qwen-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_multiblimp_multiplechoice --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-multiblimp-qwen-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_wikipedia_perplexity --model_id Qwen/Qwen3-8B --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-wikipedia-qwen-'$quant$lang$bit

        rm -rf '/workspace/.hf_home/hub/models--fifrio--Qwen3-8B-'$quant'-'$bit'bit-calibration-'$lang'-128samples'
    done
  done
done

rm -rf '/workspace/.hf_home/hub/models--Qwen--Qwen3-8B'

pixi run python -m multilingual_evaluation_belebele_multiplechoice --model_id CohereLabs/aya-expanse-8b --quantization_technique Unquantized --lang Unquantized --bit 32 2>&1 | tee 'log-belebele-aya-Unquantized'

languages=("English" "Indonesian" "Tamil" "Swahili" "Chinese")
bits=(4 3 2)
quantization_techniques=("tacq" "slimllm" "gptq")

for quant in "${quantization_techniques[@]}"; do
  for bit in "${bits[@]}" ; do
    for lang in "${languages[@]}" ; do
        # pixi run python -m multilingual_evaluation_include_multiplechoice --model_id CohereLabs/aya-expanse-8b --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-include-aya-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_floresplus_perplexity --model_id CohereLabs/aya-expanse-8b --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-floresplus-aya-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_mmluproxlite_generateuntil --model_id CohereLabs/aya-expanse-8b --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-mmluproxlite-aya-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_globalmmlulite_multiplechoice --model_id CohereLabs/aya-expanse-8b --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-globalmmlulite-aya-'$quant$lang$bit
        pixi run python -m multilingual_evaluation_belebele_multiplechoice --model_id CohereLabs/aya-expanse-8b --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-belebele-aya-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_multiblimp_multiplechoice --model_id CohereLabs/aya-expanse-8b --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-multiblimp-aya-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_wikipedia_perplexity --model_id CohereLabs/aya-expanse-8b --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-wikipedia-aya-'$quant$lang$bit

        rm -rf '/workspace/.hf_home/hub/models--fifrio--aya-expanse-8b-'$quant'-'$bit'bit-calibration-'$lang'-128samples'
    done
  done
done

rm -rf '/workspace/.hf_home/hub/models--CohereLabs--aya-expanse-8b'


pixi run python -m multilingual_evaluation_belebele_multiplechoice --model_id meta-llama/Llama-3.1-8B-Instruct --quantization_technique Unquantized --lang Unquantized --bit 32 2>&1 | tee 'log-belebele-llama-Unquantized'

languages=("English" "Indonesian" "Tamil" "Swahili" "Chinese")
bits=(4 3 2)
quantization_techniques=("tacq" "slimllm" "gptq")

for quant in "${quantization_techniques[@]}"; do
  for bit in "${bits[@]}" ; do
    for lang in "${languages[@]}" ; do
        # pixi run python -m multilingual_evaluation_include_multiplechoice --model_id meta-llama/Llama-3.1-8B-Instruct --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-include-llama-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_floresplus_perplexity --model_id meta-llama/Llama-3.1-8B-Instruct --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-floresplus-llama-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_mmluproxlite_generateuntil --model_id meta-llama/Llama-3.1-8B-Instruct --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-mmluproxlite-llama-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_globalmmlulite_multiplechoice --model_id meta-llama/Llama-3.1-8B-Instruct --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-globalmmlulite-llama-'$quant$lang$bit
        pixi run python -m multilingual_evaluation_belebele_multiplechoice --model_id meta-llama/Llama-3.1-8B-Instruct --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-belebele-llama-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_multiblimp_multiplechoice --model_id meta-llama/Llama-3.1-8B-Instruct --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-multiblimp-llama-'$quant$lang$bit
        # pixi run python -m multilingual_evaluation_wikipedia_perplexity --model_id meta-llama/Llama-3.1-8B-Instruct --quantization_technique "$quant" --lang "$lang" --bit "$bit" --nsamples 128 2>&1 | tee 'log-wikipedia-llama-'$quant$lang$bit

        rm -rf '/workspace/.hf_home/hub/models--fifrio--Llama-3.1-8B-Instruct-'$quant'-'$bit'bit-calibration-'$lang'-128samples'
    done
  done
done

rm -rf '/workspace/.hf_home/hub/models--meta-llama--Llama-3.1-8B-Instruct'