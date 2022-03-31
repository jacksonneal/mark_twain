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

# Set to use GPUS [0, 1]
GPUS = 0
# Used to run trainer in a debug mode, incurs a performance hit
DETECT_ANOMALY = True
# Run a wandb sweep
RUN_SWEEP = False
# Name used for wandb sweeps
WANDB_SWEEP_NAME = "mark_twain_sweep"
# wandb runs to explore
WANDB_COUNT = 10


def run(config=None):
    gc.collect()
    torch.cuda.empty_cache()
    pprint.pprint(config)
    try:
        seed_everything(seed=42)
        data_module = NumeraiDataModule(feature_set=config['feature_set'],
                                        sample_4th_era=config['sample_4th_era'],
                                        aux_target_cols=config['aux_target_cols'],
                                        batch_size=config['batch_size'])
        model = NumeraiModel(feature_set=config['feature_set'],
                             aux_target_cols=config['aux_target_cols'],
                             dropout=config['dropout'],
                             initial_bn=config['initial_bn'],
                             learning_rate=config['learning_rate'],
                             wd=config['wd'])
        model_summary_callback = ModelSummary(max_depth=25)
        checkpoint_callback = ModelCheckpoint(monitor="val/sharpe", mode="max")
        early_stopping_callback = EarlyStopping("val_loss", patience=3, mode="min")
        callbacks = [model_summary_callback, checkpoint_callback, early_stopping_callback]
        trainer = Trainer(detect_anomaly=DETECT_ANOMALY,
                          gpus=GPUS,
                          max_epochs=config['max_epochs'],
                          callbacks=callbacks)
        if RUN_SWEEP:
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
    if RUN_SWEEP:
        del wandb_logger
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


def run_sweep(config=None):
    with wandb.init(project=WANDB_SWEEP_NAME, config=config):
        config = wandb.config
        run(config)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    if RUN_SWEEP:
        with open('sweep.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
            wandb.login()
            sweep_id = wandb.sweep(config, project=WANDB_SWEEP_NAME)
            wandb.agent(sweep_id, function=run_sweep, count=2)
    else:
        with open('single.yaml') as f:
            config = yaml.load(f, Loader=SafeLoader)
            run(config)
