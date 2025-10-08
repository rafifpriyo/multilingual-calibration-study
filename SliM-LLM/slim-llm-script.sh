#!/bin/bash

cp ./run.py ./slim-llm
cp ./datautils.py ./slim-llm
cp ./auto.py ./AutoGPTQ/auto_gptq/modeling
cp ./__init__.py ./AutoGPTQ/auto_gptq

%cd SliM-LLM/slim-llm

pixi run python run.py \
 "google/gemma-3-1b-pt" flores 4bit --dataset_subset Indonesian --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 "google/gemma-3-1b-pt" flores 4bit --dataset_subset Tamil --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 "google/gemma-3-1b-pt" flores 4bit --dataset_subset Chinese --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 "google/gemma-3-1b-pt" flores 8bit --dataset_subset Indonesian --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 "google/gemma-3-1b-pt" flores 8bit --dataset_subset Tamil --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512

pixi run python run.py \
 "google/gemma-3-1b-pt" flores 8bit --dataset_subset Chinese --groupsize 128 \
--device "cuda" --save --seed 1234 --nsamples 512