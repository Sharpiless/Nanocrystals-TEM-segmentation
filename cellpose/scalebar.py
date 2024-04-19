import cv2
import easyocr
import numpy as np

scalebar_ocr = easyocr.Reader(['en'])

def get_scale(image, scale=None):
    black_mask = (image.sum(-1) == 0).astype(np.uint8) * 255
    H, W = black_mask.shape[:2]
    contours, hierarchy = cv2.findContours(
        black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    image_vis = image.copy()
    black_mask_vis = cv2.merge([black_mask, black_mask, black_mask])
    # 文字识别
    if scale is None:
        # 找到最大区域并填充
        area = []
        for j in range(len(contours)):
            x1 = contours[j][:, :, 0].min()
            x2 = contours[j][:, :, 0].max()
            y1 = contours[j][:, :, 1].min()
            y2 = contours[j][:, :, 1].max()
            area.append(x2 - x1)
        max_idx = np.argmax(area)
        for k in range(len(contours)):
            if k != max_idx:
                cv2.fillPoly(black_mask_vis, [contours[k]], 0)
        x1 = contours[max_idx][:, :, 0].min()
        x2 = contours[max_idx][:, :, 0].max()
        y1 = contours[max_idx][:, :, 1].min()
        y2 = contours[max_idx][:, :, 1].max()
        pix_len = x2 - x1
        image_vis = cv2.rectangle(
            image_vis, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 0, 255), 3
        )
        X1 = int(max(x1 - 0.1 * W, 0))
        X2 = int(min(x2 + 0.1 * W, W))
        Y1 = int(max(y1 - 0.1 * H, 0))
        Y2 = int(min(y2 + 0.1 * H, H))
        image_vis = cv2.rectangle(image_vis, (X1, Y1), (X2, Y2), (0, 255, 0), 3)
        word_part = image[Y1:Y2, X1:X2]
        result = scalebar_ocr.readtext(word_part)
        # result = ocr.ocr(word_part, cls=True)
        word_part_bbox = np.expand_dims(np.array(result[0][0]), 1).astype(np.int32)
        word_part_bbox[:, :, 0] += X1
        word_part_bbox[:, :, 1] += Y1
        image_vis = cv2.polylines(image_vis, [word_part_bbox], True, (0, 255, 0), 3)

        try:
            scale = eval(result[0][1].split("n")[0].strip())
        except Exception as e:
            try:
                scale = eval(result[0][1].split("u")[0].strip()) * 1000
            except Exception as e:
                print("e")
                scale = "unknown"
            
    return scale, pix_len, image_vis, [x1, y1, x2, y2]