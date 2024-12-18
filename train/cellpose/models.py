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
    """Main model which combines SizeModel and CellposeModel.

    Args:
        gpu (bool, optional): Whether or not to use GPU, will check if GPU available. Defaults to False.
        model_type (str, optional): Model type. "cyto"=cytoplasm model; "nuclei"=nucleus model; 
            "cyto2"=cytoplasm model with additional user images; 
            "cyto3"=super-generalist model; Defaults to "cyto3".
        device (torch device, optional): Device used for model running / training. Overrides gpu input. Recommended if you want to use a specific GPU (e.g. torch.device("cuda:1")). Defaults to None.

    Attributes:
        device (torch device): Device used for model running / training.
        gpu (bool): Flag indicating if GPU is used.
        diam_mean (float): Mean diameter for cytoplasm model.
        cp (CellposeModel): CellposeModel instance.
        pretrained_size (str): Pretrained size model path.
        sz (SizeModel): SizeModel instance.

    """

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

        # size model not used for bacterial model
        self.pretrained_size = size_model_path(model_type)
        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)
        self.sz.model_type = model_type

    def eval(self, x, batch_size=8, channels=[0,0], channel_axis=None, invert=False,
             normalize=True, diameter=30., do_3D=False, **kwargs):
        """Run cellpose size model and mask model and get masks.

        Args:
            x (list or array): List or array of images. Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array.
            batch_size (int, optional): Number of 224x224 patches to run simultaneously on the GPU. Can make smaller or bigger depending on GPU memory usage. Defaults to 8.
            channels (list, optional): List of channels, either of length 2 or of length number of images by 2. First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue). Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue). For instance, to segment grayscale images, input [0,0]. To segment images with cells in green and nuclei in blue, input [2,3]. To segment one grayscale image and one image with cells in green and nuclei in blue, input [[0,0], [2,3]]. Defaults to [0,0].
            channel_axis (int, optional): If None, channels dimension is attempted to be automatically determined. Defaults to None.
            invert (bool, optional): Invert image pixel intensity before running network (if True, image is also normalized). Defaults to False.
            normalize (bool, optional): If True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel; can also pass dictionary of parameters (see CellposeModel for details). Defaults to True.
            diameter (float, optional): If set to None, then diameter is automatically estimated if size model is loaded. Defaults to 30..
            do_3D (bool, optional): Set to True to run 3D segmentation on 4D image input. Defaults to False.

        Returns:
            tuple containing
                - masks (list of 2D arrays or single 3D array): Labelled image, where 0=no masks; 1,2,...=mask labels.
                - flows (list of lists 2D arrays or list of 3D arrays): 
                    - flows[k][0] = XY flow in HSV 0-255
                    - flows[k][1] = XY flows at each pixel
                    - flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics)
                    - flows[k][3] = final pixel locations after Euler integration
                - styles (list of 1D arrays of length 256 or single 1D array): Style vector summarizing each image, also used to estimate size of objects in image.
                - diams (list of diameters or float): List of diameters or float (if do_3D=True).

        """

        tic0 = time.time()
        models_logger.info(f"channels set to {channels}")

        diam0 = diameter[0] if isinstance(diameter, (np.ndarray, list)) else diameter
        estimate_size = True if (diameter is None or diam0 == 0) else False

        if estimate_size and self.pretrained_size is not None and not do_3D and x[
                0].ndim < 4:
            tic = time.time()
            models_logger.info("~~~ ESTIMATING CELL DIAMETER(S) ~~~")
            diams, _ = self.sz.eval(x, channels=channels, channel_axis=channel_axis,
                                    batch_size=batch_size, normalize=normalize,
                                    invert=invert)
            diameter = None
            models_logger.info("estimated cell diameter(s) in %0.2f sec" %
                               (time.time() - tic))
            models_logger.info(">>> diameter(s) = ")
            if isinstance(diams, list) or isinstance(diams, np.ndarray):
                diam_string = "[" + "".join(["%0.2f, " % d for d in diams]) + "]"
            else:
                diam_string = "[ %0.2f ]" % diams
            models_logger.info(diam_string)
        elif estimate_size:
            if self.pretrained_size is None:
                reason = "no pretrained size model specified in model Cellpose"
            else:
                reason = "does not work on non-2D images"
            models_logger.warning(f"could not estimate diameter, {reason}")
            diams = self.diam_mean
        else:
            diams = diameter

        tic = time.time()
        models_logger.info("~~~ FINDING MASKS ~~~")
        masks, flows, styles = self.cp.eval(x, channels=channels,
                                            channel_axis=channel_axis,
                                            batch_size=batch_size, normalize=normalize,
                                            invert=invert, diameter=diams, do_3D=do_3D,
                                            **kwargs)

        models_logger.info(">>>> TOTAL TIME %0.2f sec" % (time.time() - tic0))

        return masks, flows, styles, diams


class CellposeModel():

    def __init__(self, gpu=True, pretrained_model=False, model_type=None,
                 diam_mean=30., device=None, nchan=2):
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

    def eval(self, x, batch_size=8, resample=True, channels=None, channel_axis=None,
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
            for i in tqdm(iterator):
                maski, flowi, stylei = self.eval(
                    x[i], batch_size=batch_size,
                    channels=channels[i] if channels is not None and
                    ((len(channels) == len(x) and
                      (isinstance(channels[i], list) or
                       isinstance(channels[i], np.ndarray)) and len(channels[i]) == 2))
                    else channels, channel_axis=channel_axis, z_axis=z_axis,
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
            x = transforms.convert_image(x, channels, channel_axis=channel_axis,
                                         z_axis=z_axis, do_3D=(do_3D or
                                                               stitch_threshold > 0),
                                         nchan=self.nchan)
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
            if do_3D:
                masks, p = dynamics.resize_and_compute_masks(
                    dP, cellprob, niter=niter, cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold, interp=interp, do_3D=do_3D,
                    min_size=min_size, resize=None,
                    device=self.device if self.gpu else None)
            else:
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

class SizeModel():
    """ 
    Linear regression model for determining the size of objects in image
    used to rescale before input to cp_model.
    Uses styles from cp_model.

    Attributes:
        pretrained_size (str): Path to pretrained size model.
        cp (UnetModel or CellposeModel): Model from which to get styles.
        device (torch device): Device used for model running / training 
            (torch.device("cuda") or torch.device("cpu")), overrides gpu input,
            recommended if you want to use a specific GPU (e.g. torch.device("cuda:1")).
        diam_mean (float): Mean diameter of objects.
        
    Methods:
        eval(self, x, channels=None, channel_axis=None, normalize=True, invert=False,
             augment=False, tile=True, batch_size=8, progress=None, interp=True):
            Use images x to produce style or use style input to predict size of objects in image.

    Raises:
        ValueError: If no pretrained cellpose model is specified, cannot compute size.
    """

    def __init__(self, cp_model, device=None, pretrained_size=None, **kwargs):
        super(SizeModel, self).__init__(**kwargs)
        """ 
        Initialize size model.

        Args:
            cp_model (UnetModel or CellposeModel): Model from which to get styles.
            device (torch device, optional): Device used for model running / training 
                (torch.device("cuda") or torch.device("cpu")), overrides gpu input,
                recommended if you want to use a specific GPU (e.g. torch.device("cuda:1")).
            pretrained_size (str): Path to pretrained size model.
        """

        self.pretrained_size = pretrained_size
        self.cp = cp_model
        self.device = self.cp.device
        self.diam_mean = self.cp.diam_mean
        if pretrained_size is not None:
            self.params = np.load(self.pretrained_size, allow_pickle=True).item()
            self.diam_mean = self.params["diam_mean"]
        if not hasattr(self.cp, "pretrained_model"):
            error_message = "no pretrained cellpose model specified, cannot compute size"
            models_logger.critical(error_message)
            raise ValueError(error_message)

    def eval(self, x, channels=None, channel_axis=None, normalize=True, invert=False,
             augment=False, tile=True, batch_size=8, progress=None):
        """Use images x to produce style or use style input to predict size of objects in image.

        Object size estimation is done in two steps:
        1. Use a linear regression model to predict size from style in image.
        2. Resize image to predicted size and run CellposeModel to get output masks.
           Take the median object size of the predicted masks as the final predicted size.

        Args:
            x (list, np.ndarry): can be list of 2D/3D/4D images, or array of 2D/3D/4D images
            channels (list, optional): list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].
                Defaults to None.
            channel_axis (int, optional): channel axis in element of list x, or of np.ndarray x. 
                if None, channels dimension is attempted to be automatically determined. Defaults to None.
            normalize (bool, optional): if True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel; 
                can also pass dictionary of parameters (all keys are optional, default values shown): 
                    - "lowhigh"=None : pass in normalization values for 0.0 and 1.0 as list [low, high] (if not None, all following parameters ignored)
                    - "sharpen"=0 ; sharpen image with high pass filter, recommended to be 1/4-1/8 diameter of cells in pixels
                    - "normalize"=True ; run normalization (if False, all following parameters ignored)
                    - "percentile"=None : pass in percentiles to use as list [perc_low, perc_high]
                    - "tile_norm"=0 ; compute normalization in tiles across image to brighten dark areas, to turn on set to window size in pixels (e.g. 100)
                    - "norm3D"=False ; compute normalization across entire z-stack rather than plane-by-plane in stitching mode.
                Defaults to True.
            invert (bool, optional): Invert image pixel intensity before running network (if True, image is also normalized). Defaults to False.
            augment (bool, optional): tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
            tile (bool, optional): tiles image to ensure GPU/CPU memory usage limited (recommended). Defaults to True.
            batch_size (int, optional): number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage). Defaults to 8.
            progress (QProgressBar, optional): pyqt progress bar. Defaults to None.


        Returns:
            A tuple containing:
                - diam (np.ndarray): Final estimated diameters from images x or styles style after running both steps.
                - diam_style (np.ndarray): Estimated diameters from style alone.
        """

        if isinstance(x, list):
            diams, diams_style = [], []
            nimg = len(x)
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                diam, diam_style = self.eval(
                    x[i], channels=channels[i] if
                    (channels is not None and len(channels) == len(x) and
                     (isinstance(channels[i], list) or
                      isinstance(channels[i], np.ndarray)) and
                     len(channels[i]) == 2) else channels, channel_axis=channel_axis,
                    normalize=normalize, invert=invert, augment=augment, tile=tile,
                    batch_size=batch_size, progress=progress)
                diams.append(diam)
                diams_style.append(diam_style)

            return diams, diams_style

        if x.squeeze().ndim > 3:
            models_logger.warning("image is not 2D cannot compute diameter")
            return self.diam_mean, self.diam_mean

        styles = self.cp.eval(x, channels=channels, channel_axis=channel_axis,
                              normalize=normalize, invert=invert, augment=augment,
                              tile=tile, batch_size=batch_size, resample=False,
                              compute_masks=False)[-1]

        diam_style = self._size_estimation(np.array(styles))
        diam_style = self.diam_mean if (diam_style == 0 or
                                        np.isnan(diam_style)) else diam_style

        masks = self.cp.eval(
            x, compute_masks=True, channels=channels, channel_axis=channel_axis,
            normalize=normalize, invert=invert, augment=augment, tile=tile,
            batch_size=batch_size, resample=False,
            rescale=self.diam_mean / diam_style if self.diam_mean > 0 else 1,
            diameter=None, interp=False)[0]

        diam = utils.diameters(masks)[0]
        diam = self.diam_mean if (diam == 0 or np.isnan(diam)) else diam
        return diam, diam_style

    def _size_estimation(self, style):
        """ linear regression from style to size 
        
            sizes were estimated using "diameters" from square estimates not circles; 
            therefore a conversion factor is included (to be removed)
        
        """
        szest = np.exp(self.params["A"] @ (style - self.params["smean"]).T +
                       np.log(self.diam_mean) + self.params["ymean"])
        szest = np.maximum(5., szest)
        return szest
