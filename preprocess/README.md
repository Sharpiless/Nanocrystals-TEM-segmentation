# <p>  <b>Sophon </b> </p>

A generalist algorithm for nanocrystal segmentation in transmission electron microscopy (TEM) images (v1.0).

## Installation

Please first refer to [Training Installation](train/README.md) for installation instructions.

Then, run:

```
cd preprocess
pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0rc4
pip install -v -e .
```

Check whether the installation is successful:

```
python run.py
```