import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 
import csv
import re
import cv2
import torch.optim as optim

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset, BlindPixelTestDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR

import lightning.pytorch as pl
import torch.nn.functional as F
from utils.schedulers import LinearWarmupCosineAnnealingLR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(inp_channels=1, out_channels=1, decoder=True)
        self.loss_fn  = nn.L1Loss()
    
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
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]



def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))



def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one, 4 for blind-pixel test')

    parser.add_argument('--denoise_path', type=str, default="test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="model.ckpt", help='checkpoint save path')
    parser.add_argument('--dataset_path', type=str, default='/home/student_server/Qtt/NAFNet/data', help='absolute dataset root path')
    testopt = parser.parse_args()
    
    

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)


    ckpt_path = "ckpt/" + testopt.ckpt_name


    
    denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]

    denoise_tests = []
    derain_tests = []

    # use absolute dataset path for test inputs
    base_path = os.path.join(testopt.dataset_path, 'test_blur')
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path,i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)


    print("CKPT name : {}".format(ckpt_path))

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # infer expected input/output channels from checkpoint weights
    in_ch = None
    out_ch = None
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            # weight shape: (embed_dim, in_ch, kh, kw)
            in_ch = v.shape[1]
        if k.endswith('.output.weight'):
            # output conv weight shape: (out_ch, in_ch, kh, kw)
            out_ch = v.shape[0]

    if in_ch is None:
        in_ch = 3
    if out_ch is None:
        out_ch = in_ch

    # instantiate model matching checkpoint channel dims
    net = PromptIR(inp_channels=in_ch, out_channels=out_ch, decoder=True)

    # adapt checkpoint keys (strip lightning/prefixes) and load
    model_sd = net.state_dict()
    new_sd = {}
    for k, v in state_dict.items():
        newk = k
        if newk.startswith('net.'):
            newk = newk[len('net.'):]
        if newk.startswith('model.'):
            newk = newk[len('model.'):]
        if newk in model_sd:
            new_sd[newk] = v
            continue
        # try to match by suffix
        parts = newk.split('.')
        for i in range(1, len(parts)):
            cand = '.'.join(parts[i:])
            if cand in model_sd:
                new_sd[cand] = v
                break

    missing, unexpected = net.load_state_dict(new_sd, strict=False)
    if len(missing) > 0:
        print('Missing keys when loading checkpoint:', missing)
    if len(unexpected) > 0:
        print('Unexpected keys when loading checkpoint:', unexpected)

    net = net.cuda()
    net.eval()

    def evaluate_blind_metrics(gt_dir, output_dir, input_dir=None, mask_dir=None, save_dir='.'):
        out_imgs = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
        gt_map = {}
        for root, _, files in os.walk(gt_dir):
            for f in files:
                if f.endswith('.png'):
                    gt_map[f] = os.path.join(root, f)

        def load_blind_coords(csv_path):
            if not csv_path or not os.path.exists(csv_path):
                return None
            coords = []
            with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
                    return None
                for row in reader:
                    try:
                        coords.append((int(float(row['x'])), int(float(row['y']))))
                    except Exception:
                        continue
            if len(coords) == 0:
                return None
            arr = np.unique(np.array(coords, dtype=np.int32), axis=0)
            return arr

        def find_mask_for_image(mask_dir, img_name):
            if mask_dir is None:
                return None
            # try direct csv name
            p1 = os.path.join(mask_dir, img_name + '.csv')
            if os.path.exists(p1):
                return p1
            # try subfolder named by image
            p2 = os.path.join(mask_dir, img_name, 'blind_pixel_coords.csv')
            if os.path.exists(p2):
                return p2
            # try searching for csv with matching basename in mask_dir
            for root, _, files in os.walk(mask_dir):
                for f in files:
                    if f.endswith('.csv') and os.path.splitext(f)[0] == img_name:
                        return os.path.join(root, f)
            # fallback: if there is a single csv under mask_dir (e.g., test_mask/001.csv)
            csvs = [os.path.join(root, f) for root, _, files in os.walk(mask_dir) for f in files if f.endswith('.csv')]
            if len(csvs) == 1:
                return csvs[0]
            return None

        blind_abs_sum = 0.0
        blind_sq_sum = 0.0
        blind_abs_in_sum = 0.0
        blind_sq_in_sum = 0.0
        blind_pix_sum = 0
        per_image_logs = []

        for img_name in out_imgs:
            out_path = os.path.join(output_dir, img_name)
            gt_path = gt_map.get(img_name)
            if gt_path and os.path.exists(out_path):
                out_img = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_img is None or out_img is None:
                    continue
                if out_img.shape != gt_img.shape:
                    out_img = cv2.resize(out_img, (gt_img.shape[1], gt_img.shape[0]))

                full_psnr = float(peak_signal_noise_ratio(gt_img, out_img, data_range=255))
                full_ssim = float(structural_similarity(gt_img, out_img, data_range=255))

                row = {
                    'image': img_name,
                    'psnr': full_psnr,
                    'ssim': full_ssim,
                    'blind_mae': None,
                    'blind_rmse': None,
                    'blind_psnr': None,
                    'blind_mae_input': None,
                    'blind_mae_gain_abs': None,
                    'blind_mae_gain_pct': None,
                    'blind_count': 0
                }

                # per-image blind coords lookup
                mask_csv_for_img = find_mask_for_image(mask_dir, os.path.splitext(img_name)[0])
                blind_coords = load_blind_coords(mask_csv_for_img)
                if blind_coords is not None:
                    h, w = gt_img.shape[:2]
                    x = blind_coords[:, 0]
                    y = blind_coords[:, 1]
                    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                    if np.any(valid):
                        x = x[valid]
                        y = y[valid]
                        gt_vals = gt_img[y, x].astype(np.float64)
                        out_vals = out_img[y, x].astype(np.float64)
                        err = out_vals - gt_vals

                        blind_abs = np.abs(err)
                        blind_sq = err ** 2

                        blind_abs_sum += float(blind_abs.sum())
                        blind_sq_sum += float(blind_sq.sum())
                        blind_pix_sum += int(len(err))

                        in_mae = None
                        if input_dir is not None:
                            in_path = os.path.join(input_dir, img_name)
                            if os.path.exists(in_path):
                                in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
                                if in_img is not None:
                                    if in_img.shape != gt_img.shape:
                                        in_img = cv2.resize(in_img, (gt_img.shape[1], gt_img.shape[0]))
                                    in_vals = in_img[y, x].astype(np.float64)
                                    in_err = in_vals - gt_vals
                                    in_abs = np.abs(in_err)
                                    in_sq = in_err ** 2
                                    blind_abs_in_sum += float(in_abs.sum())
                                    blind_sq_in_sum += float(in_sq.sum())
                                    in_mae = float(in_abs.mean())

                        row.update({
                            'blind_mae': float(blind_abs.mean()),
                            'blind_rmse': float(np.sqrt(blind_sq.mean())),
                            'blind_psnr': float(10.0 * np.log10((255.0 * 255.0) / max(float(blind_sq.mean()), 1e-12))),
                            'blind_mae_input': in_mae,
                            'blind_count': int(len(err))
                        })
                        if in_mae is not None:
                            row['blind_mae_gain_abs'] = in_mae - row['blind_mae']
                            row['blind_mae_gain_pct'] = 100.0 * row['blind_mae_gain_abs'] / (in_mae + 1e-12)

                per_image_logs.append(row)

        save_blind_dir = os.path.join(save_dir, 'blind_eval')
        os.makedirs(save_blind_dir, exist_ok=True)
        save_blind_csv = os.path.join(save_blind_dir, 'test_blind_metrics.csv')
        if len(per_image_logs) > 0:
            keys = ['image', 'psnr', 'ssim', 'blind_mae', 'blind_rmse', 'blind_psnr', 'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count']
            with open(save_blind_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in per_image_logs:
                    writer.writerow(row)
            print(f"Per-image test metrics saved to: {save_blind_csv}")

        if blind_pix_sum > 0:
            blind_mae = blind_abs_sum / blind_pix_sum
            blind_mse = blind_sq_sum / blind_pix_sum
            blind_rmse = float(np.sqrt(blind_mse))
            blind_psnr = float(10.0 * np.log10((255.0 * 255.0) / max(blind_mse, 1e-12)))
            print("===> Blind-Pixel Focused Metrics")
            print(f"BlindCount(total sampled): {blind_pix_sum}")
            print(f"Blind MAE: {blind_mae:.6f} | Blind RMSE: {blind_rmse:.6f} | Blind PSNR: {blind_psnr:.3f}")

        return

    def test_Blind(net, dataset, output_subdir='blind'):
        output_path = os.path.join(testopt.output_path, output_subdir)
        subprocess.check_output(['mkdir', '-p', output_path])

        testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

        with torch.no_grad():
            for ([name], degrad_patch, _dummy) in tqdm(testloader):
                degrad_patch = degrad_patch.cuda()

                # If model expects multiple input channels but dataset gives single-channel,
                # replicate channels to match model input.
                try:
                    model_first_param = next(net.parameters())
                    model_in_ch = model_first_param.shape[1]
                except StopIteration:
                    model_in_ch = degrad_patch.shape[1]

                if degrad_patch.shape[1] != model_in_ch:
                    degrad_patch = degrad_patch.repeat(1, model_in_ch, 1, 1)

                restored = net(degrad_patch)

                # If model outputs multi-channel but target dataset is single-channel,
                # reduce output to single channel by averaging channels.
                if restored.shape[1] > 1:
                    restored_single = restored.mean(dim=1, keepdim=True)
                else:
                    restored_single = restored

                save_image_tensor(restored_single, os.path.join(output_path, name[0] + '.png'))

        print(f'Blind-pixel outputs saved to: {output_path}')

    
    if testopt.mode == 0:
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)
            out_dir = testopt.output_path + 'denoise/' + str(15) + '/'
            gt_dir = os.path.join(testopt.dataset_path, 'test_sharp')
            input_dir = os.path.join(testopt.dataset_path, 'test_blur')
            mask_dir = os.path.join(testopt.dataset_path, 'test_mask')
            # Skipping blind-pixel evaluation for BSD68 denoise outputs (name mismatch possible)
            # evaluate_blind_metrics(gt_dir, out_dir, input_dir=input_dir, mask_dir=mask_dir, save_dir=testopt.output_path)
            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)
            out_dir = testopt.output_path + 'denoise/' + str(25) + '/'
            # evaluate_blind_metrics(gt_dir, out_dir, input_dir=input_dir, mask_dir=mask_dir, save_dir=testopt.output_path)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)
            out_dir = testopt.output_path + 'denoise/' + str(50) + '/'
            # evaluate_blind_metrics(gt_dir, out_dir, input_dir=input_dir, mask_dir=mask_dir, save_dir=testopt.output_path)
    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")
    elif testopt.mode == 2:
        print('Start testing SOTS...')
        derain_base_path = testopt.derain_path
        name = derain_splits[0]
        testopt.derain_path = os.path.join(derain_base_path,name)
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
        test_Derain_Dehaze(net, derain_set, task="SOTS_outdoor")
    elif testopt.mode == 4:
        # Blind-pixel dataset testing: use test_blur (inputs), test_sharp (gt), test_mask (per-image csvs)
        print('Start blind-pixel dataset testing...')
        blind_root = os.path.join(testopt.dataset_path, 'test_blur')
        blind_set = BlindPixelTestDataset(testopt, root=blind_root)
        test_Blind(net, blind_set, output_subdir='blind')

        out_dir = os.path.join(testopt.output_path, 'blind')
        gt_dir = os.path.join(testopt.dataset_path, 'test_sharp')
        input_dir = os.path.join(testopt.dataset_path, 'test_blur')
        mask_dir = os.path.join(testopt.dataset_path, 'test_mask')
        evaluate_blind_metrics(gt_dir, out_dir, input_dir=input_dir, mask_dir=mask_dir, save_dir=testopt.output_path)
    elif testopt.mode == 3:
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)



        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:

            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")