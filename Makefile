# Makefile for mark_twain Deep Learning project

# Custom Training configurations
# ------------------------------
# File used for single run
single_conf = "single.yaml"
# File used for sweep run
sweep_conf = "sweep.yaml"
# Name used for wandb sweep
sweep_name = "mark_twain_sweep_final"
# number of wandb runs to explore during sweep
sweep_count = 30
# Workers used for parallel processing
num_workers = 8
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

tb-logs:
	python -m numerai.log --num_workers ${num_workers}

sweep:
	python -m numerai --run_sweep --sweep_name ${sweep_name} --sweep_count ${sweep_count} --num_workers ${num_workers} \
		--config ${sweep_conf}

sweep-gpu:
	python -m numerai --run_sweep --sweep_name ${sweep_name} --sweep_count ${sweep_count} --gpu \
		--num_workers ${num_workers} --config ${sweep_conf}

predict:
	# ckpt: path relative to log dir to checkpoint .ckpt file
	# hparams: path relative to log dir to hyperparameter .yaml file
	python -m numerai --predict $(ckpt) --config $(hparams)

predict-gpu:
	python -m numerai --gpu --predict $(ckpt) --config $(hparams)

submit:
	# model: name
	python -m numerai.submit $(model)

demo:
	python -m --predict demo/base/model.ckpt --config demo/base/hparams.yaml

demo-gpu:
	python -m numerai --gpu --predict demo/base/model.ckpt --config demo/base/hparams.yaml