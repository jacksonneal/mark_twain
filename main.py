import argparse

import torch
import wandb
import yaml
from yaml import SafeLoader

import constants
from trainer import MarkTwainTrainer

parser = argparse.ArgumentParser(description="Run trainer")
parser.add_argument('--gpu', action='store_true', default=False)
parser.add_argument('--run_sweep', action='store_true', default=False)
parser.add_argument('--sweep_name', type=str, default="mark_twain_sweep")
parser.add_argument('--sweep_count', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()

constants.NUM_WORKERS = args.num_workers

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    trainer = MarkTwainTrainer(args)
    if args.run_sweep:
        with open('sweep.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
            wandb.login()
            sweep_id = wandb.sweep(config, project=args.sweep_name)
            wandb.agent(sweep_id, function=trainer.run_sweep, count=2)
    else:
        with open('single.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
            trainer.run_single(config)
