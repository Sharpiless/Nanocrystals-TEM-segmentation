# <p>  <b>Sophon </b> </p>

A generalist algorithm for nanocrystal segmentation in transmission electron microscopy (TEM) images (v1.0).

## Download

Download the trained detection model from [Google Drive](), and place it into "weights" folder:
```
├── weights
    ├── cascade_hrnet_rfla_epoch_12.pth
```

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

## Generate weak labels

Assume you have a folder "data/weak_data/images", which contains images without label.

Create an empty folder "data/weak_data/weak_labels" as follows:

```
data
├── ...
├── weak_data
    ├── images
    ├── weak_labels
├── ...
```

Then run:

```
cd preprocess
python predict.py --inputs ../data/weak_data/images --outputs ../data/weak_data/weak_labels
```

Weak labels should be generated and saved to "data/weak_data/weak_labels".