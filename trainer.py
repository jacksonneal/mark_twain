import argparse
import pprint
import traceback
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.utilities.seed import seed_everything
import gc
import wandb
from yaml import SafeLoader

from data import NumeraiDataModule
from model import NumeraiModel

parser = argparse.ArgumentParser(description="Run trainer")
parser.add_argument('--gpu', action='store_true', default=False)
parser.add_argument('--run_sweep', action='store_true', default=False)
parser.add_argument('--sweep_name', type=str)
parser.add_argument('--sweep_count', type=int)
args = parser.parse_args()


def run(run_conf=None):
    gc.collect()
    torch.cuda.empty_cache()
    pprint.pprint(run_conf)
    try:
        seed_everything(seed=42)
        data_module = NumeraiDataModule(feature_set=run_conf['feature_set'],
                                        sample_4th_era=run_conf['sample_4th_era'],
                                        aux_target_cols=run_conf['aux_target_cols'],
                                        batch_size=run_conf['batch_size'])
        model = NumeraiModel(feature_set=run_conf['feature_set'],
                             aux_target_cols=run_conf['aux_target_cols'],
                             dropout=run_conf['dropout'],
                             initial_bn=run_conf['initial_bn'],
                             learning_rate=run_conf['learning_rate'],
                             wd=run_conf['wd'])
        model_summary_callback = ModelSummary(max_depth=25)
        checkpoint_callback = ModelCheckpoint(monitor="val/sharpe", mode="max")
        early_stopping_callback = EarlyStopping("val_loss", patience=3, mode="min")
        callbacks = [model_summary_callback, checkpoint_callback, early_stopping_callback]
        gpus = 1 if args.gpu else 0
        trainer = Trainer(gpus=gpus,
                          max_epochs=run_conf['max_epochs'],
                          callbacks=callbacks)
        if args.run_sweep:
            wandb_logger = WandbLogger()
            trainer.logger = wandb_logger
        trainer.fit(model, datamodule=data_module)
    except Exception as e:
        print(e)
        traceback.print_exc()
    del model
    del model_summary_callback
    del checkpoint_callback
    del early_stopping_callback
    if args.run_sweep:
        del wandb_logger
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


def run_sweep(run_conf=None):
    with wandb.init(project=args.sweep_name, config=run_conf):
        run_conf = wandb.config
        run(run_conf)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    if args.run_sweep:
        with open('sweep.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
            wandb.login()
            sweep_id = wandb.sweep(config, project=args.sweep_name)
            wandb.agent(sweep_id, function=run_sweep, count=2)
    else:
        with open('single.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
            run(config)
