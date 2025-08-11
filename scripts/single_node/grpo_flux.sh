#!/bin/bash

# This script launches the FLUX.1 training on a single node.

# CPU
#CUDA_VISIBLE_DEVICES="" accelerate launch scripts/train_flux.py --config=config/flux.py

# 1 GPU
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29502 scripts/train_flux.py --config config/flux.py

# 4 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29502 scripts/train_flux.py --config config/flux.py