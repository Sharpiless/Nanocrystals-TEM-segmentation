import cv2
import numpy as np
from run import get_bboxes

def process(image_path, bboxes):
    image = cv2.imread(image_path, 1)  # 保持原始格式的方式读取图像
    kernel = np.ones((3, 3), np.uint8)  
    masks = np.zeros_like(image)[:, :, 0]
    # image_info[key] = [np.round(float(sco_), 3), English]
    for x1, y1, x2, y2, lbl, score in bboxes:
        tgt = image[y1:y2, x1:x2]
        mask = seg_target(tgt)
        mask = cv2.medianBlur(mask, 3)
        masks[y1:y2, x1:x2] = 255 - mask
        
    dilation = cv2.dilate(masks, kernel, iterations = 1)
    erosion = cv2.erode(masks, kernel, iterations = 1)
    
    weak_mask = dilation/255 + erosion/255 == 1
    weak_mask = weak_mask.astype(np.uint8) * 255

    return np.hstack([weak_mask, masks])

def seg_target(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    media = cv2.medianBlur(gray, 3)
    _, res = cv2.threshold(media, 0, 255, cv2.THRESH_OTSU)
    return res
   

def predict_debug(x, img_path):
    global img_y
    print('predicting...', x)
    bboxes = get_bboxes(x)
    img_y = process(x, bboxes)    
    cv2.imwrite(img_path, img_y)

if __name__ == '__main__':
    # 测试单张图片并展示结果
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=str, required=True, help="loss scale for orientation")
    parser.add_argument('--outputs', type=str, required=True, help="loss scale for orientation")
    opt = parser.parse_args()
    if not os.path.exists(opt.outputs):
        print(f'mkdir {opt.outputs}.' )
        os.mkdir(opt.outputs)
    
    sub_files = os.listdir(opt.inputs)
    for img in sub_files:
        if img.endswith('.png') or img.endswith('.tif') or img.endswith('.jpg'):
            img_path = os.path.join(opt.inputs, img)
            predict_debug(img_path, os.path.join(opt.outputs, img[:-4]+".png"))

    