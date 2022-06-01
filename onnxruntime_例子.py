# ---------------------
# Bilibili: 随风而息
# Time：2022/5/25 23:20
#----------------------

import cv2
import numpy as np
import onnxruntime
import torch
import mss
from myToolkit import letterbox, non_max_suppression, scale_coords

'''
onnxruntime引擎例子
'''

cap = mss.mss()
def grab_screen_mss(monitor):
    return cv2.cvtColor(np.array(cap.grab(monitor)), cv2.COLOR_BGRA2BGR)
if __name__ == '__main__':
    # 参数定义
    img_size = (640, 640)  # 训练权重的传入尺寸
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 根据pytorch是否支持gpu选择设备
    half = (device != 'cpu')  # 如果支持cuda将使用半精度fp16,
    conf_thres = 0.25  # 置信度
    iou_thres = 0.45  # iou

    # 加载onnx引擎
    onnx_path = f'cf_v6.onnx'  # onnx模型的路径
    sess = onnxruntime.InferenceSession(onnx_path)  # 加载模型

    # # 加载单张图片
    # img_path = 'imags_14.jpg'  # 图片路径
    # img0 = cv2.imread(img_path)  # cv2读取图片
 

    scr = mss.mss() # 实例化mss
    game_left, game_tap, game_x, game_y = 0, 0, 1920, 1080  # 截图范围
    monitor = {
    'left': game_left,  # 起始点
    'top': game_tap,  # 起始点
    'width': game_x,  # 长度
    'height': game_y } # 高度       
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)    # 创建窗口
    cv2.resizeWindow('img', 1920//3, 1080//3)    # 裁剪窗口
    while True:
        if not cv2.getWindowProperty('img', cv2.WND_PROP_VISIBLE):
            cv2.destroyAllWindows()
            exit('程序结束...')
            break
        # 截图
        img0 = grab_screen_mss(monitor)
 
        # 预处理
        img = letterbox(img0, img_size, stride=64, auto=False)[0]
        # 转tensor
        img = img.transpose((2, 0, 1))[::-1]  # HWC 转 CHW，BGR 转 RGB
        # 返回一个连续的array，其内存是连续的。
        img = np.ascontiguousarray(img)
        # 放入设备
        img = torch.from_numpy(img).to(device)
        # uint8 转 fp16/32
        img = img.half() if half else img.float()
        # 归一化
        img /= 255
        # 扩大批量调暗
        if len(img.shape):
            img = img[None]

        # 推理
        img = img.cpu().numpy()  # 传入cpu并转成numpy格式
        pred_onnx = torch.tensor(sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: img})[0])

        # nms
        pred = non_max_suppression(pred_onnx, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)

        # 转换
        aims = []
        for i, det in enumerate(pred):
            if len(det):
                # 将坐标 (xyxy) 从 img_shape 重新缩放为 img0_shape
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):  # 从末尾遍历
                    # 将xyxy合并至一个维度,锚框的左上角和右下角
                    xyxy = (torch.tensor(xyxy).view(1, 4)).view(-1)
                    # 将类别和坐标合并
                    line = (cls, *xyxy)
                    # 提取tensor类型里的坐标数据
                    aim = ('%g ' * len(
                        line)).rstrip() % line  # %g 格式为浮点数 .rstrip()删除tring字符串末尾的指定字符,默认为空白符包括空格,即删除2个坐标之间的空格
                    # 划分元素
                    aim = aim.split(' ')  # 将一个元素按空格符分为多个元素,获得单个目标信息列表
                    # 所有目标的类别和锚框的坐标(类别,左上角x,左上角y,右下角x,右下角y)
                    aims.append(aim)  # 添加至列表
                print(aims)

                # 绘制
                for det in aims:
                    _, x, y, x0, y0 = det  # 每个框左上角坐标和右下角坐标,类型为str
                    cv2.rectangle(img0, (int(x), int(y)), (int(x0), int(y0)), (0, 255, 0), thickness=2)
                    # (左上角坐标,右下角坐标),str转为int    # (0, 255, 0)代表框的颜色    # 代表线条粗细,最低为1

        cv2.imshow('img', img0)
        cv2.waitKey(1)
