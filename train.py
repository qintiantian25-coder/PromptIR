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
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


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
        self.log('val_psnr', psnr)
        self.log('val_ssim', ssim)
        return {'psnr': psnr, 'ssim': ssim}
    
    def on_validation_epoch_end(self):
        # Lightning automatically computes average of logged metrics
        pass
    
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
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

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
