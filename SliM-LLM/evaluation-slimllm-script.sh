#!/bin/bash

pixi run python -m multilingual_evaluation_slimllm --lang English --bit 4

pixi run python -m multilingual_evaluation_slimllm --lang Indonesian --bit 4

pixi run python -m multilingual_evaluation_slimllm --lang Tamil --bit 4

pixi run python -m multilingual_evaluation_slimllm --lang Swahili --bit 4

pixi run python -m multilingual_evaluation_slimllm --lang Chinese --bit 4

pixi run python -m multilingual_evaluation_slimllm --lang English --bit 8

pixi run python -m multilingual_evaluation_slimllm --lang Indonesian --bit 8

pixi run python -m multilingual_evaluation_slimllm --lang Tamil --bit 8

pixi run python -m multilingual_evaluation_slimllm --lang Swahili --bit 8

pixi run python -m multilingual_evaluation_slimllm --lang Chinese --bit 8