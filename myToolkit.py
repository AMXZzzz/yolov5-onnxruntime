# ---------------------
# Bilibili: 随风而息
# Time：2022/5/25 23:20

import torchvision
import cv2
import numpy as np
import torch

'''
工具包,所有的预处理和后处理函数都在这
'''

'''坐标转换'''
# 中心点xy,宽高wh转换到左上角xy和右下角xy
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# 左上角xy,右下角xy转换至左上角xy,宽高wh
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

'''--------预处理--------'''
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # 在满足跨步约束的同时调整图像大小和填充图像
    shape = im.shape[:2]  # 当前形状[高度，宽度]  #注意顺序
    # print(f'图片尺寸:{shape}')
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)  # new_shape:(640, 640)

    # 计算图片尺寸的比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) # 找出图片的(高宽)最小的比例，[0]是高，[1]是宽
    if not scaleup:  # 只缩小，不放大（为了更好的 val mAP）默认跳过
        r = min(r, 1.0) # # 若有大于1的则用1比例，若有小于1的则选最小，更新r

    # 计算填充边缘
    ratio = r, r  # 高宽比，用上面计算的最小r作为宽高比
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # round:四舍五入,宽高,注意顺序 new_unpad:(640, 361)
    # print(f'按比例需要缩放到:{new_unpad}')
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding (640-640) ,(640-361)
    # print(f'填充的大小,dw:{dw},dh:{dh}')
    if auto:  # 最小矩形,为False
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding   # mod:计算两数组对应位置元素的余数。
        # print(f'最小矩形dw:{dw},dh:{dh}')
    elif scaleFill:  # 缩放，一般为False
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    dw /= 2  # 将填充分为 2 条边
    dh /= 2
    if shape[::-1] != new_unpad:  #  裁剪 shape[::-1]:(1376, 776) new_unpad:(640, 361)
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))    # 防止过填充
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # print(f'填充后的图片尺寸:{im.shape}')
    return im, ratio, (dw, dh)

'''------后处理------'''
# NMS
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        max_det=300):
    """

    返回：检测列表，每个图像的 (n,6) tensor [xyxy, conf, cls]   # [左上角坐标xy右下角坐标xy,置信度,类别]
    """

    nc = prediction.shape[2]- 5 # 类别数量
    # print(prediction.shape)
    xc = prediction[..., 4] > conf_thres  # 候选框

    # 设置
    min_wh, max_wh = 2, 4096   # （像素）最小和最大盒子宽度和高度
    max_nms = 3000  # torchvision.ops.nms() 中的最大框数
    multi_label &= nc > 1  # 每个候选框的多标签设置（增加 0.5ms/img）



    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # 图像索引xi，图像推断x    # enumerate可遍历的数据对象组合为一个索引序列
        # 应用约束
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # 宽高
        x = x[xc[xi]]  # 置信度

        # 如果没有数据就处理下一个图像
        if not x.shape[0]:
            continue

        # 计算配置
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # 框（中心x，中心y，宽度，高度）到（x1，y1，x2，y2）
        box = xywh2xyxy(x[:, :4])

        # 检测矩阵 nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # 只有最好类
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # 按类别过滤
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # 框的个数
        if not n:  # 没有锚框
            continue
        elif n > max_nms:  # 多余的锚框
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 按置信度排序

        # 批量NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别
        boxes, scores = x[:, :4] + c, x[:, 4]  # 框（类别偏移），分数
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # 极限检测
            i = i[:max_det]

        output[xi] = x[i]

    return output


def clip_coords(boxes, shape):
    # 将边界 xyxy 框裁剪为图像形状（高度、宽度）
    if isinstance(boxes, torch.Tensor):  # tensor类型
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array 类型
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

# 转换
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # 将坐标 (xyxy) 从 img_shape 重新缩放为 img0_shape
    if ratio_pad is None:  # 从 img0_shape 计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 比例 = 旧 / 新
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh 填充大小
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]  # x 填充
    coords[:, [1, 3]] -= pad[1]  # y 填充
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
