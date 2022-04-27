# Makefile for mark_twain Deep Learning project

# Custom Training configurations
# ------------------------------
# File used for single run
single_conf = "single.yaml"
# File used for sweep run
sweep_conf = "sweep.yaml"
# Name used for wandb sweep
sweep_name = "mark_twain_sweep_milestone_2"
# number of wandb runs to explore during sweep
sweep_count = 2
# Workers used for parallel processing
num_workers = 14
# ------------------------------

SHELL := /bin/bash

req-gpu:
	srun --partition=gpu --nodes=1 --pty --gres=gpu:v100-sxm2:1 --ntasks=1 --mem=8GB --time=08:00:00 /bin/bash

init:
	pip install -r requirements.txt

single:
	python -m numerai --num_workers ${num_workers} --config ${single_conf}

single-gpu:
	python -m numerai --gpu --num_workers ${num_workers} --config ${single_conf}

sweep:
	python -m numerai --run_sweep --sweep_name ${sweep_name} --sweep_count ${sweep_count} --num_workers ${num_workers} \
		--config ${sweep_conf}

sweep-gpu:
	python -m numerai --run_sweep --sweep_name ${sweep_name} --sweep_count ${sweep_count} --gpu \
		--num_workers ${num_workers} --config ${sweep_conf}

predict:
	python -m numerai --predict $(ckpt) --config $(hparams)

tb-logs:
	python -m numerai/log --num_workers ${num_workers}