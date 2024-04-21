# <p>  <b>Sophon </b> </p>

A generalist algorithm for nanocrystal segmentation in transmission electron microscopy (TEM) images (v1.0).

Software is available at [Baidu Disk (提取码：gbnt)](https://pan.baidu.com/s/1Q_3n1sHRCqzB56Sz54JSqw) and [Google Drive](https://drive.google.com/file/d/1-klszlNdlVBu8xbInkTYjnsokVglEOtr/view?usp=drive_link).

![Software](../assets/software.png)

![Segmentation](../assets/1-0001_visualization.png)

## System requirements

The software has been heavily tested on Windows 10.

## Basic usage

The software is built basically on [Cellpose](https://github.com/MouseLand/cellpose).

For basic usage, you can refer to [Cellpose Docs](https://cellpose.readthedocs.io/en/latest/gui.html#using-the-gui).

We show the different parts in the following.

## Calibration

Our modified software support automatically calibrate the scale using modern OCR tools.

![Calibration](../assets/cailbration1.png)

![Calibration](../assets/cailbration2.png)

You can also input manually in the text box.

## Run segmentation

We provide the checkpoints of a robust segmentation model (**epoch28.pth**).

Select the checkpoints and click **[run]**.

![Calibration](../assets/run1.png)

![Calibration](../assets/run2.png)

## Export visualization and measurement data

Our software also supports exporting segmentation visualization and measurement data.

![Calibration](../assets/export.png)

These results will be saved to the same folder with the input image.

![Calibration](../assets/export2.png)

The pixel-wise and real mesurements are saved to the ".csv" file:

![Calibration](../assets/measurement.png)

## Train your own model

Please refer to [Training Installation](train/README.md) for installation instructions.

## Deploy your own model

First, add the model to GUI by clicking "Add custom torch model to GUI" in "Models".

![load custom model to GUI](../assets/load_model.png)

Then, select the model by clicking "custom models" in "Other models".

![load custom model to GUI](../assets/load_model2.png)