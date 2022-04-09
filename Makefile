# Makefile for mark_twain Deep Learning project

# Custom Training configurations
# ------------------------------
# Name used for wandb sweep
sweep_name = "mark_twain_sweep_milestone_2"
# number of wandb runs to explore during sweep
sweep_count = 2
# Workers used for parallel processing
num_workers = 28
# ------------------------------

gpu:
	srun --partition=gpu --nodes=1 --pty --gres=gpu:v100-sxm2:1 --ntasks=1 --mem=4GB --time=08:00:00 /bin/bash

source:
	source /etc/profile.d/modules.sh

anaconda: source
	module load anaconda3/2022.01

cuda: source
	module load cuda/11.1

env: anaconda cuda
	source activate pytorch_env_training

single:
	python main.py --num_workers ${num_workers}

single-gpu:
	python main.py --gpu --num_workers ${num_workers}

sweep:
	python main.py --run_sweep --sweep_name ${sweep_name} --sweep_count ${sweep_count} --num_workers ${num_workers}

sweep-gpu:
	python main.py --run_sweep --sweep_name ${sweep_name} --sweep_count ${sweep_count} --gpu --num_workers ${num_workers}

tb-logs:
	python tb_logs.py --num_workers ${num_workers}