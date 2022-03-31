# Makefile for mark_twain Deep Learning project

# Custom Training configurations
# ------------------------------
# Name used for wandb sweep
sweep_name = "mark_twain_sweep"
# number of wandb runs to explore during sweep
sweep_count = 2
# ------------------------------

single:
	python trainer.py

single-gpu:
	python trainer.py --gpu

sweep:
	python trainer.py --run_sweep --sweep_name ${sweep_name} --sweep_count ${sweep_count}

sweep-gpu:
	python trainer.py --run_sweep --sweep_name ${sweep_name} --sweep_count ${sweep_count} --gpu

tb-logs:
	python tb_logs.py
