#!/bin/bash

pixi run python -m multilingual_evaluation_unquantized

pixi run python -m multilingual_evaluation_llmint8 --bit 4

pixi run python -m multilingual_evaluation_llmint8 --bit 8