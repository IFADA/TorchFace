import numpy as np
import torch


def iou(boxe, boxes, isMin=False):
    top_x = np.maximum(boxe[0], boxes[:, 0])
    top_y = np.maximum(boxe[1], boxes[:, 1])
    bottom_x = np.minimum(boxe[2], boxes[:, 2])
    bottom_y = np.minimum(boxe[3], boxes[:, 3])
    boxe_area = (boxe[2] - boxe[0]) * (boxe[3] - boxe[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    j_area = np.maximum(0, (bottom_x - top_x) * (bottom_y - top_y))
    if isMin:
        # 交集/极小值（应用于O网络）
        fm = np.minimum(boxes_area, boxe_area)
    else:
        #交集/并集（应用于PR网络）
        fm = boxe_area + boxes_area - j_area
    # print('iou',j_area/fm)
    return j_area / fm


def nms(boxes, thresh=0.3, isMin=False):
    # print('nms',boxes.shape)
    if boxes.shape[0] == 0:
        return np.array([])
    # 所有数据按照置信度从大到小排序
    _boxes = boxes[(-boxes[:, 4]).argsort()]

    r_boxes = []
    while _boxes.shape[0] > 1:  # ？？？？？？？？
        # 置信度最大的框
        a_box = _boxes[0]
        # 存放置信度最大的
        r_boxes.append(a_box)
        # 剩余的图片
        rest_boxes = _boxes[1:]
        # if rest_boxes.shape[0] == 0:
        #     break
        index = np.where(iou(a_box, rest_boxes, isMin) < thresh)
        _boxes = rest_boxes[index]
    # print('r_bo',r_boxes)
    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])
    return np.stack(r_boxes)


def convert_to_squre(bbox):
    squre_bbox = bbox.copy()

    if squre_bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    squre_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5  # x1的坐标 -x1 side/2 + max side/2
    squre_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    squre_bbox[:, 2] = squre_bbox[:, 0] + max_side
    squre_bbox[:, 3] = squre_bbox[:, 1] + max_side

    return squre_bbox
