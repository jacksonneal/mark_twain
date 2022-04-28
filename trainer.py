import pprint
import traceback
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.utilities.seed import seed_everything
import gc
import wandb

from data import NumeraiDataModule
from lit import NumeraiLit


class MarkTwainTrainer:
    def __init__(self, args):
        self.args = args

    def run_single(self, run_conf=None):
        gc.collect()
        torch.cuda.empty_cache()
        pprint.pprint(run_conf)
        try:
            seed_everything(seed=42)
            data_module = NumeraiDataModule(feature_set=run_conf['feature_set'],
                                            sample_4th_era=run_conf['sample_4th_era'],
                                            aux_target_cols=run_conf['aux_target_cols'],
                                            batch_size=run_conf['batch_size'],
                                            num_workers=self.args.num_workers)
            model = NumeraiLit(model_name=run_conf['model_name'],
                               feature_set=run_conf['feature_set'],
                               dimensions=run_conf['dimensions'],
                               aux_target_cols=run_conf['aux_target_cols'],
                               dropout=run_conf['dropout'],
                               initial_bn=run_conf['initial_bn'],
                               learning_rate=run_conf['learning_rate'],
                               wd=run_conf['wd'])
            model_summary_callback = ModelSummary(max_depth=25)
            checkpoint_callback = ModelCheckpoint(monitor="val/sharpe", mode="max")
            early_stopping_callback = EarlyStopping("val_loss", patience=10000, mode="min")
            callbacks = [model_summary_callback, checkpoint_callback, early_stopping_callback]
            gpus = 1 if self.args.gpu else 0
            trainer = Trainer(gpus=gpus,
                              max_epochs=run_conf['max_epochs'],
                              callbacks=callbacks)
            if self.args.run_sweep:
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
        if self.args.run_sweep:
            del wandb_logger
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    def run_sweep(self, run_conf=None):
        with wandb.init(project=self.args.sweep_name, config=run_conf):
            run_conf = wandb.config
            self.run_single(run_conf)
