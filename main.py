import argparse
import os

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from helpers.dataset import setup_dataloaders
from module import CycleGAN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_modality", required=True)
    parser.add_argument("--target_modality", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    args = parse_args()

    cyclegan = CycleGAN()
    trn_dataloader, val_dataloader, tst_dataloader = setup_dataloaders(dataset_h5py='translation_mbrats_cyclegan.h5',
                                                                       A_key=args.source_modality,
                                                                       B_key=args.target_modality,
                                                                       batch_size=8,
                                                                       num_workers=max(0, os.cpu_count()))

    logger = TensorBoardLogger(save_dir=f'logs/{args.source_modality}-{args.target_modality}',
                               name=cyclegan.__class__.__name__)

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{logger.experiment.log_dir}/checkpoints",
            every_n_epochs=5,
            filename='{epoch}',
            save_last=True,
            save_top_k = -1
        ),
    ]

    trainer = pl.Trainer(precision=32,
                         accelerator='auto',
                         max_epochs=200,
                         callbacks=callbacks,
                         logger=logger,
                         limit_train_batches=10,
                         limit_val_batches=10
                         )
    trainer.fit(model=cyclegan,
                train_dataloaders=trn_dataloader,
                val_dataloaders=val_dataloader)
