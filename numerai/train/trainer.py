import os
import pprint
import traceback
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.utilities.seed import seed_everything
import gc
import wandb
from numerai.data.data_module import NumeraiDataModule
from numerai.definitions import LOG_DIR, WANDB_LOG_DIR, PREDICTIONS_CSV, ROOT_DIR
from numerai.model.lit import NumeraiLit


class MarkTwainTrainer:
    def __init__(self, args):
        self.args = args

    def run_single(self, run_conf=None, ckpt=None):
        gc.collect()
        torch.cuda.empty_cache()
        pprint.pprint(run_conf)
        try:
            seed_everything(seed=42)
            data_module = NumeraiDataModule(feature_set=run_conf['feature_set'],
                                            sample_4th_era=run_conf['sample_4th_era'],
                                            aux_target_cols=run_conf['aux_target_cols'],
                                            batch_size=run_conf['batch_size'],
                                            num_workers=self.args.num_workers,
                                            pca=run_conf['pca'])
            data_module.prepare_data()
            model = NumeraiLit(model_name=run_conf['model_name'],
                               feature_set=run_conf['feature_set'],
                               num_features=data_module.num_features,
                               dimensions=run_conf['dimensions'],
                               aux_target_cols=run_conf['aux_target_cols'],
                               dropout=run_conf['dropout'],
                               initial_bn=run_conf['initial_bn'],
                               learning_rate=run_conf['learning_rate'],
                               wd=run_conf['wd'],
                               kernel=run_conf['kernel'],
                               stride=run_conf['stride'],
                               pool_kernel=run_conf[
                                   'pool_kernel']) if ckpt is None else NumeraiLit.load_from_checkpoint(
                checkpoint_path=os.path.join(ROOT_DIR, ckpt))

            model_summary_callback = ModelSummary(max_depth=25)
            callbacks = [model_summary_callback]
            if ckpt is None:
                checkpoint_callback = ModelCheckpoint(monitor="val/spearman", mode="max", save_weights_only=True)
                early_stopping_callback = EarlyStopping("train_loss", patience=5, mode="min")
                callbacks += [checkpoint_callback, early_stopping_callback]

            gpus = 1 if self.args.gpu else 0
            max_epochs = run_conf['max_epochs'] if ckpt is None else 0
            trainer = Trainer(gpus=gpus,
                              max_epochs=max_epochs,
                              callbacks=callbacks)

            logger = WandbLogger(save_dir=WANDB_LOG_DIR) if self.args.run_sweep else TensorBoardLogger(save_dir=LOG_DIR)
            trainer.logger = logger

            if ckpt is None:
                trainer.fit(model, datamodule=data_module)
            else:
                print('Predicting...')
                predictions = trainer.predict(model, datamodule=data_module)
                print('Completed predictions')
                if run_conf['model_name'] == 'AEMLP' or run_conf['model_name'] == 'CAE':
                    predictions = list(map(lambda preds: preds[2], predictions))
                predictions = torch.cat(predictions).squeeze()
                if len(run_conf['aux_target_cols']) > 0:
                    predictions = predictions[:, 0]
                out_df = data_module.test_data.df
                out_df.loc[:, "prediction"] = predictions
                print('Saving predictions...')
                out_df["prediction"].to_csv(PREDICTIONS_CSV)
                print('Predictions saved!')

        except Exception as e:
            print(e)
            traceback.print_exc()
        del model
        del model_summary_callback
        if ckpt is None:
            del checkpoint_callback
            del early_stopping_callback
        del logger
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    def run_sweep(self, run_conf=None):
        with wandb.init(project=self.args.sweep_name, config=run_conf, dir=WANDB_LOG_DIR):
            run_conf = wandb.config
            self.run_single(run_conf)
