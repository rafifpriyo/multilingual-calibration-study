#!/bin/bash

pixi run python -m multilingual_evaluation_gptq --lang English --bit 4

pixi run python -m multilingual_evaluation_gptq --lang Indonesian --bit 4

pixi run python -m multilingual_evaluation_gptq --lang Tamil --bit 4

pixi run python -m multilingual_evaluation_gptq --lang Swahili --bit 4

pixi run python -m multilingual_evaluation_gptq --lang Chinese --bit 4

pixi run python -m multilingual_evaluation_gptq --lang English --bit 8

pixi run python -m multilingual_evaluation_gptq --lang Indonesian --bit 8

pixi run python -m multilingual_evaluation_gptq --lang Tamil --bit 8

pixi run python -m multilingual_evaluation_gptq --lang Swahili --bit 8

pixi run python -m multilingual_evaluation_gptq --lang Chinese --bit 8