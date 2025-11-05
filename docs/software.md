# <p>  <b>Sophon </b> </p>

A generalist algorithm for nanocrystal segmentation in transmission electron microscopy (TEM) images (v1.0).



https://github.com/user-attachments/assets/b16d4648-ace7-4285-81e3-0c6b2e63e904



Download the software from https://github.com/Sharpiless/Nanocrystals-TEM-segmentation/releases/tag/v2.1

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

Or you can input manually in the text box. Press **Enter** after inputing.

![Calibration](../assets/cailbration3.png)

Here is the calibration results (automatical):

![Calibration](../assets/cailbration2.png)

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
