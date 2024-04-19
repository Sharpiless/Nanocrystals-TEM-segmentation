"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
from pathlib import Path
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import torch

import logging

models_logger = logging.getLogger(__name__)

from . import transforms, dynamics, utils, plot
from .resnet_torch import CPnet
from .core import assign_device, check_mkl, run_net, run_3D
import cv2
import matplotlib.pyplot as plt
from .plot import mask_overlay

_MODEL_URL = "https://www.cellpose.org/models"
_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = pathlib.Path.home().joinpath(".cellpose", "models")
MODEL_DIR = pathlib.Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT

MODEL_NAMES = [
    "cyto3", "nuclei", "cyto2_cp3", "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3",
    "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3", "cyto2", "cyto"
]

MODEL_LIST_PATH = os.fspath(MODEL_DIR.joinpath("gui_models.txt"))

normalize_default = {
    "lowhigh": None,
    "percentile": None,
    "normalize": True,
    "norm3D": False,
    "sharpen_radius": 0,
    "smooth_radius": 0,
    "tile_norm_blocksize": 0,
    "tile_norm_smooth3D": 1,
    "invert": False
}

def show_segmentation(img, maski, save_path):
    
    plt.cla()
    plt.clf()
    fig = plt.figure(figsize=(10, 3))
    
    ax = fig.add_subplot(1, 3, 1)
    img0 = img.copy()

    ax.imshow(img0)
    ax.set_title("original image")
    ax.axis("off")

    outlines = utils.masks_to_outlines(maski)

    overlay = mask_overlay(img0, maski)

    ax = fig.add_subplot(1, 3, 2)
    outX, outY = np.nonzero(outlines)
    imgout = img0.copy()
    imgout[outX, outY] = np.array([255, 0, 0])  # pure red

    ax.imshow(imgout)
    ax.set_title("predicted outlines")
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(overlay)
    ax.set_title("predicted masks")
    ax.axis("off")
    
    fig.savefig(save_path, dpi=300)
    plt.close()

def mask2measurement(mask):
    # 获取所有独立粒子的标识符（忽略背景0）
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # 假设背景为0，移除背景标签

    result_dict = {
        "label": [],
        "area": [],
        "perimeter": [],
        "equivalent_diameter": [],
        "max_diameter": [],
        "min_diameter": [],
    }
    # 遍历每个粒子的标识符
    for label in unique_labels:
        # 为当前粒子创建掩码
        particle_mask = np.uint8(mask == label)
        # 找到当前粒子的轮廓
        contours, _ = cv2.findContours(
            particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # 应对找到多个轮廓的情况，正常情况下应该只有一个轮廓对应一个粒子
        if contours:
            # 假设每个标签只对应一个粒子，故取第一个轮廓
            contour = contours[0]
            # 计算面积
            area = cv2.contourArea(contour)
            # 计算周长
            perimeter = cv2.arcLength(contour, True)
            # 计算等效直径
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 计算矩形的长和宽（最大和最小直径）
            width, height = rect[1]
            max_diameter = max(width, height)
            min_diameter = min(width, height)

            result_dict["label"].append(label)
            result_dict["area"].append(area)
            result_dict["perimeter"].append(perimeter)
            result_dict["equivalent_diameter"].append(equivalent_diameter)
            result_dict["max_diameter"].append(max_diameter)
            result_dict["min_diameter"].append(min_diameter)
    return result_dict

def model_path(model_type, model_index=0):
    torch_str = "torch"
    if model_type == "cyto" or model_type == "cyto2" or model_type == "nuclei":
        basename = "%s%s_%d" % (model_type, torch_str, model_index)
    else:
        basename = model_type
    return cache_model_path(basename)


def size_model_path(model_type):
    torch_str = "torch"
    if model_type == "cyto" or model_type == "nuclei" or model_type == "cyto2":
        basename = "size_%s%s_0.npy" % (model_type, torch_str)
    else:
        basename = "size_%s.npy" % model_type
    return cache_model_path(basename)


def cache_model_path(basename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_MODEL_URL}/{basename}"
    cached_file = os.fspath(MODEL_DIR.joinpath(basename))
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        utils.download_url_to_file(url, cached_file, progress=True)
    return cached_file


def get_user_models():
    model_strings = []
    if os.path.exists(MODEL_LIST_PATH):
        with open(MODEL_LIST_PATH, "r") as textfile:
            lines = [line.rstrip() for line in textfile]
            if len(lines) > 0:
                model_strings.extend(lines)
    return model_strings


class Cellpose():

    def __init__(self, gpu=False, model_type="cyto3", device=None):
        super(Cellpose, self).__init__()

        # assign device (GPU or CPU)
        sdevice, gpu = assign_device(use_torch=True, gpu=gpu)
        self.device = device if device is not None else sdevice
        self.gpu = gpu

        model_type = "cyto3" if model_type is None else model_type

        self.diam_mean = 30.  #default for any cyto model
        nuclear = "nuclei" in model_type
        if nuclear:
            self.diam_mean = 17.

        self.cp = CellposeModel(device=self.device, gpu=self.gpu, model_type=model_type,
                                diam_mean=self.diam_mean)
        self.cp.model_type = model_type

    def eval(self, x, batch_size=8, channel_axis=None, invert=False,
             normalize=True, diameter=30., do_3D=False, **kwargs):
        tic0 = time.time()
        diams = diameter
        models_logger.info("~~~ FINDING MASKS ~~~")
        masks, flows, styles = self.cp.eval(x,
                                            channel_axis=channel_axis,
                                            batch_size=batch_size, normalize=normalize,
                                            invert=invert, diameter=diams, do_3D=do_3D,
                                            **kwargs)

        models_logger.info(">>>> TOTAL TIME %0.2f sec" % (time.time() - tic0))

        return masks, flows, styles, diams


class CellposeModel():

    def __init__(self, gpu=False, pretrained_model=False, model_type=None,
                 diam_mean=30., device=None, nchan=3):
        self.diam_mean = diam_mean
        builtin = True

        if model_type is not None or (pretrained_model and
                                      not os.path.exists(pretrained_model)):
            pretrained_model_string = model_type if model_type is not None else "cyto"
            model_strings = get_user_models()
            all_models = MODEL_NAMES.copy()
            all_models.extend(model_strings)
            if ~np.any([pretrained_model_string == s for s in MODEL_NAMES]):
                builtin = False
            elif ~np.any([pretrained_model_string == s for s in all_models]):
                pretrained_model_string = "cyto3"

            if (pretrained_model and not os.path.exists(pretrained_model[0])):
                models_logger.warning("pretrained model has incorrect path")
            models_logger.info(f">> {pretrained_model_string} << model set to be used")

            if pretrained_model_string == "nuclei":
                self.diam_mean = 17.
            else:
                self.diam_mean = 30.
            pretrained_model = model_path(pretrained_model_string)

        else:
            builtin = False
            if pretrained_model:
                pretrained_model_string = pretrained_model
                models_logger.info(f">>>> loading model {pretrained_model_string}")

        # assign network device
        self.mkldnn = None
        if device is None:
            sdevice, gpu = assign_device(use_torch=True, gpu=gpu)
        self.device = device if device is not None else sdevice
        if device is not None:
            device_gpu = self.device.type == "cuda"
        self.gpu = gpu if device is None else device_gpu
        if not self.gpu:
            self.mkldnn = check_mkl(True)

        # create network
        self.nchan = nchan
        self.nclasses = 3
        nbase = [32, 64, 128, 256]
        self.nbase = [nchan, *nbase]
        
        self.net = CPnet(self.nbase, self.nclasses, sz=3, mkldnn=self.mkldnn,
                         max_pool=True, diam_mean=diam_mean).to(self.device)

        self.pretrained_model = pretrained_model
        if self.pretrained_model:
            self.net.load_model(self.pretrained_model, device=self.device)
            if not builtin:
                self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
            self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]
            models_logger.info(
                f">>>> model diam_mean = {self.diam_mean: .3f} (ROIs rescaled to this size during training)"
            )
            if not builtin:
                models_logger.info(
                    f">>>> model diam_labels = {self.diam_labels: .3f} (mean diameter of training ROIs)"
                )

        self.net_type = "cellpose"

    def eval(self, x, batch_size=8, resample=True, channel_axis=None,
             z_axis=None, normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             stitch_threshold=0.0, min_size=15, niter=None, augment=False, tile=True,
             tile_overlap=0.1, bsize=224, interp=True, compute_masks=True,
             progress=None):
        if isinstance(x, list) or x.squeeze().ndim == 5:
            masks, styles, flows = [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                maski, flowi, stylei = self.eval(
                    x[i], batch_size=batch_size,
                    channel_axis=channel_axis, z_axis=z_axis,
                    normalize=normalize, invert=invert,
                    rescale=rescale[i] if isinstance(rescale, list) or
                    isinstance(rescale, np.ndarray) else rescale,
                    diameter=diameter[i] if isinstance(diameter, list) or
                    isinstance(diameter, np.ndarray) else diameter, do_3D=do_3D,
                    anisotropy=anisotropy, augment=augment, tile=tile,
                    tile_overlap=tile_overlap, bsize=bsize, resample=resample,
                    interp=interp, flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold, compute_masks=compute_masks,
                    min_size=min_size, stitch_threshold=stitch_threshold,
                    progress=progress, niter=niter)
                masks.append(maski)
                flows.append(flowi)
                styles.append(stylei)
            return masks, flows, styles

        else:
            # reshape image
            x = transforms.convert_image(x)
            if x.ndim < 4:
                x = x[np.newaxis, ...]
            self.batch_size = batch_size

            if diameter is not None and diameter > 0:
                rescale = self.diam_mean / diameter
            elif rescale is None:
                diameter = self.diam_labels
                rescale = self.diam_mean / diameter

            masks, styles, dP, cellprob, p = self._run_cp(
                x, compute_masks=compute_masks, normalize=normalize, invert=invert,
                rescale=rescale, resample=resample, augment=augment, tile=tile,
                tile_overlap=tile_overlap, bsize=bsize, flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold, interp=interp, min_size=min_size,
                do_3D=do_3D, anisotropy=anisotropy, niter=niter,
                stitch_threshold=stitch_threshold)

            flows = [plot.dx_to_circ(dP), dP, cellprob, p]
            return masks, flows, styles

    def _run_cp(self, x, compute_masks=True, normalize=True, invert=False, niter=None,
                rescale=1.0, resample=True, augment=False, tile=True, tile_overlap=0.1,
                cellprob_threshold=0.0, bsize=224, flow_threshold=0.4, min_size=15,
                interp=True, anisotropy=1.0, do_3D=False, stitch_threshold=0.0):

        if isinstance(normalize, dict):
            normalize_params = {**normalize_default, **normalize}
        elif not isinstance(normalize, bool):
            raise ValueError("normalize parameter must be a bool or a dict")
        else:
            normalize_params = normalize_default
            normalize_params["normalize"] = normalize
        normalize_params["invert"] = invert

        tic = time.time()
        shape = x.shape
        nimg = shape[0]

        bd, tr = None, None

        # pre-normalize if 3D stack for stitching or do_3D
        do_normalization = True if normalize_params["normalize"] else False
        if nimg > 1 and do_normalization and (stitch_threshold or do_3D):
            # must normalize in 3D if do_3D is True
            normalize_params["norm3D"] = True if do_3D else normalize_params["norm3D"]
            x = np.asarray(x)
            x = transforms.normalize_img(x, **normalize_params)
            # do not normalize again
            do_normalization = False

        if do_3D:
            img = np.asarray(x)
            yf, styles = run_3D(self.net, img, rsz=rescale, anisotropy=anisotropy,
                                augment=augment, tile=tile, tile_overlap=tile_overlap)
            cellprob = yf[0][-1] + yf[1][-1] + yf[2][-1]
            dP = np.stack(
                (yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]),
                axis=0)  # (dZ, dY, dX)
            del yf
        else:
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            styles = np.zeros((nimg, self.nbase[-1]), np.float32)
            if resample:
                dP = np.zeros((2, nimg, shape[1], shape[2]), np.float32)
                cellprob = np.zeros((nimg, shape[1], shape[2]), np.float32)
            else:
                dP = np.zeros(
                    (2, nimg, int(shape[1] * rescale), int(shape[2] * rescale)),
                    np.float32)
                cellprob = np.zeros(
                    (nimg, int(shape[1] * rescale), int(shape[2] * rescale)),
                    np.float32)
            for i in iterator:
                img = np.asarray(x[i])
                if do_normalization:
                    img = transforms.normalize_img(img, **normalize_params)
                if rescale != 1.0:
                    img = transforms.resize_image(img, rsz=rescale)
                yf, style = run_net(self.net, img, bsize=bsize, augment=augment,
                                    tile=tile, tile_overlap=tile_overlap)
                if resample:
                    yf = transforms.resize_image(yf, shape[1], shape[2])

                cellprob[i] = yf[:, :, 2]
                dP[:, i] = yf[:, :, :2].transpose((2, 0, 1))
                if self.nclasses == 4:
                    if i == 0:
                        bd = np.zeros_like(cellprob)
                    bd[i] = yf[:, :, 3]
                styles[i][:len(style)] = style
            del yf, style
        styles = styles.squeeze()

        net_time = time.time() - tic
        if nimg > 1:
            models_logger.info("network run in %2.2fs" % (net_time))

        if compute_masks:
            tic = time.time()
            niter0 = 200 if (do_3D and not resample) else (1 / rescale * 200)
            niter = niter0 if niter is None or niter==0 else niter
            if 1:
                masks, p = [], []
                resize = [shape[1], shape[2]] if (not resample and
                                                  rescale != 1) else None
                iterator = trange(nimg, file=tqdm_out,
                                  mininterval=30) if nimg > 1 else range(nimg)
                for i in iterator:
                    outputs = dynamics.resize_and_compute_masks(
                        dP[:, i],
                        cellprob[i],
                        niter=niter,
                        cellprob_threshold=cellprob_threshold,
                        flow_threshold=flow_threshold,
                        interp=interp,
                        resize=resize,
                        min_size=min_size if stitch_threshold == 0 or nimg == 1 else
                        -1,  # turn off for 3D stitching
                        device=self.device if self.gpu else None)
                    masks.append(outputs[0])
                    p.append(outputs[1])

                masks = np.array(masks)
                p = np.array(p)
                if stitch_threshold > 0 and nimg > 1:
                    models_logger.info(
                        f"stitching {nimg} planes using stitch_threshold={stitch_threshold:0.3f} to make 3D masks"
                    )
                    masks = utils.stitch3D(masks, stitch_threshold=stitch_threshold)
                    masks = utils.fill_holes_and_remove_small_masks(
                        masks, min_size=min_size)
                elif nimg > 1:
                    models_logger.warning("3D stack used, but stitch_threshold=0 and do_3D=False, so masks are made per plane only")


            flow_time = time.time() - tic
            if nimg > 1:
                models_logger.info("masks created in %2.2fs" % (flow_time))
            masks, dP, cellprob, p = masks.squeeze(), dP.squeeze(), cellprob.squeeze(
            ), p.squeeze()

        else:
            masks, p = np.zeros(0), np.zeros(0)  #pass back zeros if not compute_masks
        return masks, styles, dP, cellprob, p
