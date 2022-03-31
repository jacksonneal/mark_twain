# Makefile for mark_twain Deep Learning project

# Custom Training configurations
# ------------------------------
# Name used for wandb sweep
sweep_name = "mark_twain_sweep"
# number of wandb runs to explore during sweep
sweep_count = 2
# Workers used for parallel processing
num_workers = 4
# ------------------------------

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