# <p>  <b>Sophon </b> </p>

A generalist algorithm for nanocrystal segmentation in transmission electron microscopy (TEM) images (v1.0).

Software is available at [Baidu Disk (提取码：gbnt)](https://pan.baidu.com/s/1Q_3n1sHRCqzB56Sz54JSqw) and [Google Drive](https://drive.google.com/file/d/1-klszlNdlVBu8xbInkTYjnsokVglEOtr/view?usp=drive_link).

![Software](assets/software.png)

![Segmentation](assets/1-0001_visualization.png)

## Train your own model

Please refer to [Training Installation](train/README.md) for installation instructions.

## Deploy your own model

First, add the model to GUI by clicking "Add custom torch model to GUI" in "Models".

![load custom model to GUI](../assets/load_model.png)

Then, select the model by clicking "custom models" in "Other models".

![load custom model to GUI](../assets/load_model2.png)

## Acknowledgments
* The code is heavily borrowed from [Cellpose](https://github.com/MouseLand/cellpose), we thank the authors for their great effort.