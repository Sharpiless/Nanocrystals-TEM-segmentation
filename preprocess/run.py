import cv2
import numpy as np
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

threshold = 0.5
# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'configs/cascade_rcnn/exp0511v2_rfla.py'
checkpoint_file = 'weights/cascade_hrnet_rfla_epoch_12.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cpu')

# 初始化可视化工具
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# 从 checkpoint 中加载 Dataset_meta，并将其传递给模型的 init_detector
visualizer.dataset_meta = model.dataset_meta
classes = model.dataset_meta['classes']

def get_bboxes(img_path):
    result = inference_detector(model, img_path)
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    pred_bboxes = []
    for i in range(bboxes.shape[0]):
        if scores[i] > threshold:
            x1, y1, x2, y2 = bboxes[i].astype(np.int64)
            lbl = classes[labels[i]]
            pred_bboxes.append((x1, y1, x2, y2, lbl, scores[i]))
    return pred_bboxes