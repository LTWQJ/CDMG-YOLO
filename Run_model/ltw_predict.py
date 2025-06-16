import os
import sys
from pathlib import Path
from shlex import split

from PIL import Image
import numpy as np
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True`'
from ultralytics import YOLO


def get_resize_size(self, img):
    _img = img.copy()
    img_height, img_width, depth = _img.shape
    ratio = img_width / img_height
    if ratio >= self.show_width / self.show_height:
        self.img_width = self.show_width
        self.img_height = int(self.img_width / ratio)
    else:
        self.img_height = self.show_height
        self.img_width = int(self.img_height * ratio)
    return self.img_width, self.img_height


# Load a model

def detect(model_path):
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/CDMG_YOLO2/weights/best.pt"   # cdmg_yolo
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/CDG_YOLO/weights/best.pt"
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/CG_YOLO/weights/best.pt"
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/CMG_YOLO/weights/best.pt"
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/DG_YOLO/weights/best.pt"
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/DMG_YOLO/weights/best.pt"
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/MG_YOLO/weights/best.pt"
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-DG-YOLO/weights/best.pt"
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-DMG-YOLO/weights/best.pt"
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-MG-YOLO/weights/best.pt"
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-CDG-YOLO/weights/best.pt"
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-CDMG-YOLO/weights/best.pt"
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-CG-YOLO/weights/best.pt"
    # model_path = "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-CMG-YOLO/weights/best.pt"
    model = YOLO(model_path)  # load an official model
    tif_path = r"E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/data/val/00044.jpg"
    dir = 'predict-levr-mine/' + tif_path.split("/")[-1].split(".")[0]
    name = model_path.split("/")[-1].split(".")[0]
    # Predict with the model
    results = model.predict(
        source=tif_path,  # 输入图像的路径,
        save=True,  # 保存预测结果
        imgsz=640,  # 输入图像的大小，可以是整数或w，
        conf=0.25,  # 用于检测的目标置信度阈值（默认为0.25，用于预测，0.001用于验证）
        iou=0.45,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        show=False,  # 如果可能的话，显示结果
        project=dir,  # (str, optional) 项目名称
        name=name,  # (str, optional) 实验名称，结果保存在'project/name'目录下
        save_txt=True,  # 保存结果为 .txt 文件
        save_conf=True,  # 保存结果和置信度分数
        save_crop=True,  # 保存裁剪后的图像和结果
        show_labels=True,  # 在图中显示目标标签
        show_conf=True,  # 在图中显示目标置信度分数
        vid_stride=1,  # 视频帧率步长
        line_width=3,  # 边界框线条粗细（像素）
        visualize=False,  # 可视化模型特征
        augment=False,  # 对预测源应用图像增强
        agnostic_nms=False,  # 类别无关的NMS
        retina_masks=True,  # 使用高分辨率的分割掩码
        boxes=True,  # 在分割预测中显示边界框
    )

if __name__ == '__main__':
    # models = ["E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/CDMG_YOLO2/weights/best.pt", "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/CDG_YOLO/weights/best.pt",
    #           "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/CG_YOLO/weights/best.pt", "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/CMG_YOLO/weights/best.pt",
    #           "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/DG_YOLO/weights/best.pt", "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/DMG_YOLO/weights/best.pt",
    #           "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/MG_YOLO/weights/best.pt", "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-DG-YOLO/weights/best.pt",
    #           "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-DMG-YOLO/weights/best.pt", "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-MG-YOLO/weights/best.pt",
    #           "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-CDG-YOLO/weights/best.pt", "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-CDMG-YOLO/weights/best.pt",
    #           "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-CG-YOLO/weights/best.pt", "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/MMDD/NoP2-CMG-YOLO/weights/best.pt"]
    models = ["E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/yolo_model/model_weigh/best-cdmg.pt",
              "E:/pycharm.2022.3.2/Workplace/YOLO11_CLASS/Mine-detection/yolo_model/model_weigh/best-yolo.pt"]
    for model in models:
        detect(model)
        print(model)