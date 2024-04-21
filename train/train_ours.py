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
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Train model on EMPS dataset.')
    parser.add_argument('--data-dir', metavar='data_dir', type=str, help='directory which contains the data.')
    parser.add_argument('--device', metavar='device', type=str, default='cuda', help='device to train on (cuda or cpu)')
    parser.add_argument('--im-size', metavar='im_size', type=tuple, default=(512, 512), help='image size to load for training.')
    parser.add_argument('--labeled_data', action='store_true', help="increase h/w gradually for smoother texture")
    parser.add_argument('--extra_data', action='store_true', help="whether to load extra dataset")
    parser.add_argument('--weak_data', action='store_true', help="whether to load weakly-supervised data")
    parser.add_argument('--epochs', type=int, default=200, help='epoches for training.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for training.')
    parser.add_argument('--save_path', type=str, help='save checkpoints locally.')
    parser.add_argument('--ckpt_path', type=str, help='load checkpoints locally.')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    train_dataset = EMPSDataset(args.data_dir, 
                                data_mod='train',
                                im_size=args.im_size, 
                                device=device,
                                labeled_data=args.labeled_data,
                                extra_data=args.extra_data,
                                weak_data=args.weak_data)

    test_dataset = EMPSDataset(args.data_dir, 
                                data_mod='test',
                                im_size=args.im_size, 
                                device=device,
                                transform=False,
                                labeled_data=True)
    
    test_data, test_labels = [], []

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    
    for (images, instances, _, _, is_full_labeled) in tqdm(test_loader):
        test_data.append(images.cpu().numpy())
        test_labels.append(instances.cpu().numpy())

    # model name and path
    initial_model = "cyto"  # @param ["cyto", "cyto3","nuclei","tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3", "scratch"]
    model_name = "CP_tissuenet"  # @param {type:"string"}

    # @markdown ###Advanced Parameters

    # @markdown ###If not, please input:
    learning_rate = 0.1  # @param {type:"number"}
    weight_decay = 0.0001  # @param {type:"number"}

    # DEFINE CELLPOSE MODEL (without size model)
    if args.ckpt_path:
        model = models.CellposeModel(gpu=True, pretrained_model=args.ckpt_path, nchan=3)
    else:
        model = models.CellposeModel(gpu=True, model_type=initial_model, nchan=3)
    network = model.net

    optimizer = Adam(network.parameters(), lr=args.lr)
    end_lr = args.lr / 5
    decay = compute_decay_rate(start_lr=args.lr, end_lr=end_lr, epochs=int(args.epochs*0.75))
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=decay)

    best_ap50 = 0.0
    for epoch in range(args.epochs):
        network.train()

        for (images, instances, segmaps, flows, is_full_labeled) in tqdm(train_loader):
            images, flows = images.to(device), flows.to(device)
            instances, segmaps = instances.to(device), segmaps.to(device)
            is_full_labeled = is_full_labeled.to(device)

            y = network(images)[0]
            if is_full_labeled.sum() > 0:
                loss = _loss_fn_seg(flows[is_full_labeled], y[is_full_labeled], device)
            else:
                loss = 0.0
            not_full_labeled = torch.logical_not(is_full_labeled)
            if not_full_labeled.sum() > 0:
                loss += _loss_fn_weak(instances[not_full_labeled], segmaps[not_full_labeled], y[not_full_labeled], device)
            else:
                loss += 0.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        masks = model.eval(test_data)[0]
        # check performance using ground truth labels
        miou, ap = metrics.average_precision(test_labels, masks)[:2]
            
        ap50 = ap[:,0].mean()
        ap75 = ap[:,1].mean()
        ap90 = ap[:,2].mean()
        miou = miou.mean()
            
        print("")
        print(f">>> epoch: {epoch:04d} AP at iou threshold 0.50 = {ap50:.3f}")
        print(f">>> epoch: {epoch:04d} AP at iou threshold 0.75 = {ap75:.3f}")
        print(f">>> epoch: {epoch:04d} AP at iou threshold 0.90 = {ap90:.3f}")
        print(f">>> epoch: {epoch:04d} mIOU = {miou:.3f}")
            
        if ap50 > best_ap50:
            best_ap50 = ap50
            print(f">>> epoch: {epoch:04d} best AP at iou threshold 0.50 = {best_ap50:.3f}")
            network.save_model(os.path.join(args.save_path, "best.pth"))
            
        lr_scheduler.step()
    
        network.save_model(os.path.join(args.save_path, "last.pth"))