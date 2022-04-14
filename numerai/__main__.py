import argparse
import os

import torch
import wandb
import yaml
from yaml import SafeLoader

from numerai.definitions import CONF_DIR
from numerai.train.trainer import MarkTwainTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run trainer")
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--run_sweep', action='store_true', default=False)
    parser.add_argument('--sweep_name', type=str, default="mark_twain_sweep")
    parser.add_argument('--sweep_count', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    torch.multiprocessing.freeze_support()
    trainer = MarkTwainTrainer(args)
    fp = os.path.join(CONF_DIR, args.config)
    with open(fp) as f:
        config = yaml.load(f, Loader=SafeLoader)
        if args.run_sweep:
            wandb.login()
            sweep_id = wandb.sweep(config, project=args.sweep_name)
            wandb.agent(sweep_id, function=trainer.run_sweep, count=args.sweep_count)
        else:
            trainer.run_single(config)


if __name__ == '__main__':
    main()
