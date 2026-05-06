import os
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset, BlindPairedTrainDataset, ValidationDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import compute_psnr_ssim
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class BestModelAnnouncer(pl.Callback):
    """Callback that prints a message when ModelCheckpoint saves a new best model."""
    def __init__(self):
        super().__init__()
        self.last_best = None

    def on_validation_end(self, trainer, pl_module):
        # find ModelCheckpoint callback
        for cb in getattr(trainer, 'callbacks', []):
            try:
                if isinstance(cb, ModelCheckpoint):
                    best_path = getattr(cb, 'best_model_path', None)
                    best_score = getattr(cb, 'best_model_score', None)
                    if best_path and best_path != self.last_best:
                        print(f"*** New best model saved: {best_path} (score={best_score})")
                        self.last_best = best_path
                    break
            except Exception:
                continue


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        self.best_psnr = 0.0
        self.best_model_path = None
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ([fname], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        psnr, ssim, _ = compute_psnr_ssim(restored, clean_patch)
        # Log aggregated metrics at epoch level so ModelCheckpoint can monitor them
        self.log('val_psnr', float(psnr), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_ssim', float(ssim), on_epoch=True, prog_bar=True, sync_dist=True)
        return {'psnr': psnr, 'ssim': ssim}

    def on_validation_epoch_start(self):
        epoch = getattr(self, 'current_epoch', None)
        print(f"--- Validation starting for epoch {epoch} ---")

    def on_validation_epoch_end(self):
        # print aggregated metrics and current best checkpoint info
        metrics = {}
        try:
            metrics['val_psnr'] = float(self.trainer.callback_metrics.get('val_psnr', float('nan')))
            metrics['val_ssim'] = float(self.trainer.callback_metrics.get('val_ssim', float('nan')))
        except Exception:
            metrics['val_psnr'] = float('nan')
            metrics['val_ssim'] = float('nan')

        # find ModelCheckpoint callback if present
        best_path = None
        best_score = None
        for cb in getattr(self.trainer, 'callbacks', []):
            try:
                from lightning.pytorch.callbacks import ModelCheckpoint as _MC
                if isinstance(cb, _MC):
                    best_path = getattr(cb, 'best_model_path', None)
                    best_score = getattr(cb, 'best_model_score', None)
                    break
            except Exception:
                continue

        print(f"--- Validation finished. val_psnr={metrics['val_psnr']:.4f}, val_ssim={metrics['val_ssim']:.4f}")
        if best_path:
            print(f"Current best checkpoint: {best_path} (score={best_score})")
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]




def main():
    print("Options")
    print(opt)
    # Always create a TensorBoard logger to generate local training logs
    tb_logger = TensorBoardLogger(save_dir="logs/")
    loggers = [tb_logger]
    if opt.wblogger is not None:
        wb_logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train")
        # keep both W&B and TensorBoard loggers
        loggers.insert(0, wb_logger)
    # if only one logger, Trainer accepts a single logger; otherwise pass list
    logger = loggers if len(loggers) > 1 else loggers[0]

    if getattr(opt, 'use_blind_pairs', False):
        print('Using BlindPairedTrainDataset from', opt.dataset_path)
        trainset = BlindPairedTrainDataset(opt, root=opt.dataset_path)
    else:
        trainset = PromptTrainDataset(opt)
    
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    # load validation dataset if paths provided
    valloader = None
    if opt.val_blur_dir and opt.val_sharp_dir:
        print(f'Loading validation dataset from {opt.val_blur_dir} and {opt.val_sharp_dir}')
        valset = ValidationDataset(opt.val_blur_dir, opt.val_sharp_dir)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
        print(f'Validation set size: {len(valset)}')
    
    model = PromptIRModel()
    model.best_model_path = opt.best_model_path

    best_model_dir = os.path.dirname(opt.best_model_path) or opt.ckpt_dir
    best_model_name = os.path.splitext(os.path.basename(opt.best_model_path))[0] or 'best_model'
    # ensure directory exists for best model
    try:
        os.makedirs(best_model_dir, exist_ok=True)
    except Exception:
        pass
    
    # only save best model based on val_psnr; automatically overwrite when psnr improves
    callbacks = []
    if valloader is not None:
        best_model_callback = ModelCheckpoint(
            dirpath=best_model_dir,
            filename=best_model_name,
            monitor='val_psnr',
            mode='max',
            save_top_k=1
        )
        callbacks.append(best_model_callback)
        # announce when best model updates
        callbacks.append(BestModelAnnouncer())
        print(f'Best model will be saved to {opt.best_model_path} (updated when PSNR improves)')
    else:
        print('Warning: No validation dataset provided; no model will be saved')
    
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=opt.val_interval,  # validate every N epochs
    )
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == '__main__':
    main()
