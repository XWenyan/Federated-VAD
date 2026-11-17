import glob
import os
import random
import time

import cv2
import numpy as np
import torch.utils.data

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]

# 定义异常检测训练数据集类，继承自PyTorch的Dataset基类
class AbnormalDatasetGradientsTrain(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        if args.dataset == "avenue":
            data_path = args.avenue_path
        elif args.dataset == "shanghai":
            data_path = args.shanghai_path
        else:
            raise Exception("Unknown dataset!")
        # 保存异常样本的比例（用于随机选择异常/正常样本）
        self.percent_abnormal = args.percent_abnormal
        # 标记是否使用3D输入（包含前后帧的拼接）
        self.input_3d = args.input_3d 
        # 调用_read_data方法读取数据，返回异常帧、正常帧、梯度图、异常掩码的路径列表
        self.abnormal_data, self.data, self.gradients, self.masks_abnormal = self._read_data(data_path)

    def _read_data(self, data_path):
        # 初始化存储数据路径的列表：正常帧、梯度图、异常帧、异常掩码
        data = []
        gradients = []
        abnormal_data = []
        masks_abnormal = []
        extension = None
        # 遍历支持的扩展名，确定数据集中实际使用的图像格式
        for ext in IMG_EXTENSIONS:
            if len(list(glob.glob(os.path.join(data_path, "train/frames", f"*/*{ext}")))) > 0:
                extension = ext
                break
        # 获取train/frames目录下的所有子目录（每个子目录对应一个视频序列）
        dirs = list(glob.glob(os.path.join(data_path, "train", "frames", "*")))
        # 遍历每个视频目录
        for dir in dirs:
            imgs_path = list(glob.glob(os.path.join(dir, f"*{extension}"))) # 获取当前视频目录下所有图像的路径
            data += imgs_path # 将当前视频的正常帧路径添加到全局正常帧列表
            video_name = os.path.basename(dir)
            gradients_path = [] # 获取视频名称（子目录的basename）
            # 为当前视频的每个帧生成对应的梯度图、异常帧、异常掩码路径
            for img_path in imgs_path:
                # 获取文件名并保持前导零
                frame_name = os.path.basename(img_path).split('.')[0]
                # 拼接梯度图路径（train/gradients2/视频名/帧编号.png）
                gradients_path.append(os.path.join(data_path, "train", "gradients2", video_name,
                                              f"{frame_name}.png"))
                # 拼接异常帧路径（train/frames_abnormal/视频名/帧编号.png）
                abnormal_data.append(os.path.join(data_path, "train", "frames_abnormal", video_name,
                                                  f"{frame_name}.png"))
                # 拼接异常掩码路径（train/masks_abnormal/视频名/帧编号.png）
                masks_abnormal.append(os.path.join(data_path, "train", "masks_abnormal", video_name,
                                                  f"{frame_name}.png"))
            # 将当前视频的梯度图路径添加到全局梯度图列表
            gradients += gradients_path
        return abnormal_data, data, gradients, masks_abnormal # 返回所有收集到的路径列表

    def __getitem__(self, index):
        random_uniform = random.uniform(0, 1) # 生成0-1之间的随机数，用于决定当前样本是异常还是正常
        if random_uniform <= self.percent_abnormal: # 若随机数小于等于异常样本比例，则选择异常帧
            # 读取异常帧图像
            img = cv2.imread(self.abnormal_data[index])
            # 提取异常帧的元信息（所在目录、帧编号、帧编号的位数，用于补零）
            dir_path, frame_no, len_frame_no = self.extract_meta_info(self.abnormal_data, index)
            # 读取前3帧（若不存在则用当前帧替代
            previous_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=-3, length=len_frame_no)
            # # 读取后3帧（若不存在则用当前帧替代）
            next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=3, length=len_frame_no)
            # 若使用3D输入，则将前帧、当前帧、后帧在通道维度拼接（形成时间维度的3帧输入）
            if self.input_3d:
                img = np.concatenate([previous_img, img, next_img], axis=-1)
            # 读取异常掩码（只取单通道，用于标记异常区域）
            mask = cv2.imread(self.masks_abnormal[index])[:,:,:1]

        else: # 否则选择正常帧
            img = cv2.imread(self.data[index]) # 读取正常帧图像
            # 提取正常帧的元信息
            dir_path, frame_no, len_frame_no = self.extract_meta_info(self.data, index)
            # 读取前3帧和后3帧
            previous_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=-3, length=len_frame_no)
            next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=3, length=len_frame_no)
            # 若使用3D输入，则拼接前帧、当前帧、后帧
            if self.input_3d:
                img = np.concatenate([previous_img, img, next_img], axis=-1)
            # 正常帧的掩码为全0（无异常区域）
            mask = np.zeros((img.shape[0],img.shape[1],1),dtype=np.uint8)
        gradient = cv2.imread(self.gradients[index]) # 读取当前帧对应的梯度图
        target = cv2.imread(self.data[index])  # 读取目标图像（正常帧，用于模型学习恢复正常特征）

        # 若图像尺寸与输入尺寸不符，则调整大小（输入尺寸由args指定）
        if img.shape[:2] != self.args.input_size or gradient.shape[:2] != self.args.input_size:
            img = cv2.resize(img, self.args.input_size[::-1]) # opencv的resize是(w,h)，需反转输入尺寸
            gradient = cv2.resize(gradient, self.args.input_size[::-1])
            mask = cv2.resize(mask, self.args.input_size[::-1])
            mask = np.expand_dims(mask, axis=-1) # 确保掩码是单通道（H,W,1）
        # 调整目标图像尺寸
        if target.shape[:2] != self.args.input_size:
            target = cv2.resize(target, self.args.input_size[::-1])

        # 将目标图像与掩码拼接（目标包含原始正常帧和异常掩码，用于监督训练）
        target = np.concatenate((target, mask), axis=-1)
        # 数据类型转换为float32
        img = img.astype(np.float32)
        gradient = gradient.astype(np.float32)
        target = target.astype(np.float32)
        # 归一化到[-1, 1]区间（图像像素值原本是0-255，(x-127.5)/127.5后范围为-1到1）
        img = (img - 127.5) / 127.5
        # 转换维度：从HWC（高度、宽度、通道）转换为CHW（通道、高度、宽度），适配PyTorch输入格式
        img = np.swapaxes(img, 0, -1).swapaxes(1, -1)
        # 目标图像同样归一化和维度转换
        target = (target - 127.5) / 127.5
        target = np.swapaxes(target, 0, -1).swapaxes(1, -1)
        # 梯度图维度转换为CHW
        gradient = np.swapaxes(gradient, 0, 1).swapaxes(0, -1)
        # 返回处理后的输入图像、梯度图、目标图像
        return img, gradient, target

    def extract_meta_info(self, data, index):
        frame_no = int(data[index].split("/")[-1].split('.')[0]) # 从数据路径中提取帧编号（文件名前缀，如"0001.png"的编号是1）
        dir_path = "/".join(data[index].split("/")[:-1]) # 提取当前帧所在的目录路径
        len_frame_no = len(data[index].split("/")[-1].split('.')[0]) # 提取帧编号的位数（用于补零，如"0001"是4位，后续构造相邻帧路径时需保持位数一致）
        return dir_path, frame_no, len_frame_no

    def read_prev_next_frame_if_exists(self, dir_path, frame_no, direction=-3, length=1):
        # 构造相邻帧的路径（direction为-3表示前3帧，+3表示后3帧；length是帧编号的位数，用于补零）
        frame_path = dir_path + "/" + str(frame_no + direction).zfill(length) + ".png"
        # 若相邻帧存在则读取，否则读取当前帧（避免路径不存在的错误）
        if os.path.exists(frame_path):
            return cv2.imread(frame_path)
        else:
            return cv2.imread(dir_path + "/" + str(frame_no).zfill(length) + ".png")

    def __len__(self):
        return len(self.data) # 数据集长度为正常帧的数量（异常帧与正常帧长度一致，通过索引对应）

    def __repr__(self):
        return self.__class__.__name__ # 打印类名（用于调试时显示数据集类型）
