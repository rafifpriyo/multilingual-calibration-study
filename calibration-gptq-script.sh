#!/bin/bash

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang English --bit 2 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang Indonesian --bit 2 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang Tamil --bit 2 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang Swahili --bit 2 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang Chinese --bit 2 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang English --bit 4 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang Indonesian --bit 4 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang Tamil --bit 4 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang Swahili --bit 4 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang Chinese --bit 4 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang English --bit 8 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang Indonesian --bit 8 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang Tamil --bit 8 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang Swahili --bit 8 --nsamples 128

pixi run python -m multilingual_calibration_gptq --model_id Qwen/Qwen3-4B --lang Chinese --bit 8 --nsamples 128