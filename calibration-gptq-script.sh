#!/bin/bash

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang English --bit 2 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang Indonesian --bit 2 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang Tamil --bit 2 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang Swahili --bit 2 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang Chinese --bit 2 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang English --bit 3 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang Indonesian --bit 3 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang Tamil --bit 3 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang Swahili --bit 3 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang Chinese --bit 3 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang English --bit 4 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang Indonesian --bit 4 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang Tamil --bit 4 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang Swahili --bit 4 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-8B --lang Chinese --bit 4 --nsamples 128