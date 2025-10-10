#!/bin/bash

cp ./run.py ./slim-llm
cp ./datautils.py ./slim-llm
cp ./auto.py ./AutoGPTQ/auto_gptq/modeling
cp ./__init__.py ./AutoGPTQ/auto_gptq

cd slim-llm

pixi run python run.py \
 Qwen/Qwen3-1.7B flores 4bit --dataset_subset English --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 Qwen/Qwen3-1.7B flores 4bit --dataset_subset Indonesian --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 Qwen/Qwen3-1.7B flores 4bit --dataset_subset Tamil --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 Qwen/Qwen3-1.7B flores 4bit --dataset_subset Swahili --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 Qwen/Qwen3-1.7B flores 4bit --dataset_subset Chinese --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 Qwen/Qwen3-1.7B flores 8bit --dataset_subset English --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 Qwen/Qwen3-1.7B flores 8bit --dataset_subset Indonesian --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 Qwen/Qwen3-1.7B flores 8bit --dataset_subset Tamil --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 Qwen/Qwen3-1.7B flores 8bit --dataset_subset Swahili --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 Qwen/Qwen3-1.7B flores 8bit --dataset_subset Chinese --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512