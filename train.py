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


class BestModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that prints a message after a new best model is saved."""
    def __init__(self, *args, log_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_reported_best = None
        self.log_file = log_file

    def _write_log(self, message):
        if not self.log_file:
            return
        try:
            with open(self.log_file, 'a', encoding='utf-8') as handle:
                handle.write(message + '\n')
        except Exception:
            pass

    def on_validation_end(self, trainer, pl_module):
        previous_best = self.best_model_path
        super().on_validation_end(trainer, pl_module)
        current_best = self.best_model_path
        current_score = self.best_model_score
        val_psnr = trainer.callback_metrics.get('val_psnr')
        val_ssim = trainer.callback_metrics.get('val_ssim')
        epoch = getattr(trainer, 'current_epoch', None)

        summary = (
            f"epoch={epoch} val_psnr={float(val_psnr):.4f} "
            f"val_ssim={float(val_ssim):.4f} best_path={current_best} best_score={current_score}"
        )
        self._write_log("[VALIDATION] " + summary)
        print("[VALIDATION] " + summary)

        if current_best and current_best != previous_best and current_best != self._last_reported_best:
            save_message = f"*** New best model saved: {current_best} (score={current_score})"
            print(save_message)
            self._write_log("[SAVE] " + save_message)
            self._last_reported_best = current_best


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
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored,clean_patch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ([fname], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        psnr, ssim, _ = compute_psnr_ssim(restored, clean_patch)
        # 去掉 sync_dist=True 以让 ModelCheckpoint 正确读取指标
        self.log('val_psnr', float(psnr), on_epoch=True, prog_bar=True)
        self.log('val_ssim', float(ssim), on_epoch=True, prog_bar=True)
        return {'psnr': psnr, 'ssim': ssim}

    def on_validation_epoch_start(self):
        epoch = getattr(self, 'current_epoch', None)
        print(f"--- Validation starting for epoch {epoch} ---")

    def on_validation_epoch_end(self):
        metrics = {}
        try:
            metrics['val_psnr'] = float(self.trainer.callback_metrics.get('val_psnr', float('nan')))
            metrics['val_ssim'] = float(self.trainer.callback_metrics.get('val_ssim', float('nan')))
        except Exception:
            metrics['val_psnr'] = float('nan')
            metrics['val_ssim'] = float('nan')

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
    tb_logger = TensorBoardLogger(save_dir="logs/")
    loggers = [tb_logger]
    if opt.wblogger is not None:
        wb_logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train")
        loggers.insert(0, wb_logger)
    logger = loggers if len(loggers) > 1 else loggers[0]

    if getattr(opt, 'use_blind_pairs', False):
        print('Using BlindPairedTrainDataset from', opt.dataset_path)
        trainset = BlindPairedTrainDataset(opt, root=opt.dataset_path)
    else:
        trainset = PromptTrainDataset(opt)
    
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
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
    try:
        os.makedirs(best_model_dir, exist_ok=True)
    except Exception:
        pass

    log_file = os.path.join(opt.ckpt_dir, 'train_status.log')
    try:
        os.makedirs(opt.ckpt_dir, exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as handle:
            handle.write('=== training session started ===\n')
    except Exception:
        log_file = None
    
    callbacks = []
    if valloader is not None:
        best_model_callback = BestModelCheckpoint(
            dirpath=best_model_dir,
            filename=best_model_name,
            monitor='val_psnr',
            mode='max',
            save_top_k=1,
            log_file=log_file,
        )
        callbacks.append(best_model_callback)
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
        check_val_every_n_epoch=opt.val_interval,
    )
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == '__main__':
    main()