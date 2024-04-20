import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
from glob import glob
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import argparse
from cellpose import train
from cellpose import models
from data import EMPSDataset
from torchvision.utils import save_image
import torch
from tqdm import tqdm
import torch.nn as nn

def compute_decay_rate(start_lr, end_lr, epochs):
    return np.exp(np.log(end_lr/start_lr)/epochs)

def _loss_fn_seg(lbl, y, device):
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    veci = 5. * lbl[:, 1:]
    loss = criterion(y[:, :2], veci)
    loss /= 2.
    loss2 = criterion2(y[:, -1], (lbl[:, 0] > 0.5).float())
    loss = loss + loss2
    return loss

def _loss_fn_weak(lbl1, lbl2, y, device):
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    valid_lbl = lbl2[lbl1 < 0.5]
    valid_y = y[:, -1][lbl1 < 0.5]
    loss2 = criterion2(valid_y, (valid_lbl > 0.5).float())
    return loss2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on EMPS dataset.')
    parser.add_argument('--data-dir', metavar='data_dir', default="data", type=str, help='Directory which contains the data.')
    parser.add_argument('--device', metavar='device', type=str, default='cuda', help='device to train on (cuda or cpu)')
    parser.add_argument('--im-size', metavar='im_size', type=tuple, default=(512, 512), help='Image size to load for training.')
    parser.add_argument('--pretrained', type=str, default=None, help='device to train on (cuda or cpu)')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = EMPSDataset(args.data_dir, 
                                data_mod='test',
                                im_size=args.im_size, 
                                device=device,
                                transform=False,
                                labeled_data=True)
    test_data, test_labels = [], []
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    for (images, instances, _, _, is_full_labeled) in tqdm(test_loader):
        test_data.append(images.squeeze().cpu().numpy())
        test_labels.append(instances.squeeze().cpu().numpy())

    # model name and path
    initial_model = "cyto"  # @param ["cyto", "cyto3","nuclei","tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3", "scratch"]
    model_name = "CP_tissuenet"  # @param {type:"string"}

    # DEFINE CELLPOSE MODEL (without size model)
    model = models.CellposeModel(gpu=True, pretrained_model=args.pretrained, nchan=3)

    masks = model.eval(test_data, bsize=args.im_size[0])[0]

    miou, ap = metrics.average_precision(test_labels, masks)[:2]
            
    ap50 = ap[:,0].mean()
    ap75 = ap[:,1].mean()
    ap90 = ap[:,2].mean()
            
    print("")
    print(f">>> average precision at iou threshold 0.50 = {ap50:.3f}")
    print(f">>> average precision at iou threshold 0.75 = {ap75:.3f}")
    print(f">>> average precision at iou threshold 0.90 = {ap90:.3f}")
    
    miou = miou.mean()
    print(f">>> mIOU = {miou:.3f}")