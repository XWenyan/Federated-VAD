import random
import numpy as np
from collections import defaultdict
import torch
import os
# 按视频目录分组
def group_indices_by_video(dataset):
    """
    将数据集的索引按视频目录分组(同一视频的帧索引归为一组)
    返回:视频组列表,每个元素是该视频包含的所有帧索引
    """
    # 构建 视频目录 -> 帧索引 的映射
    video_groups = defaultdict(list)
    for idx in range(len(dataset)):
        # 适配不同数据集的路径字段
        if hasattr(dataset, 'data') and len(dataset.data) > idx:
            frame_path = dataset.data[idx]
        elif hasattr(dataset, 'img_paths') and len(dataset.img_paths) > idx:
            frame_path = dataset.img_paths[idx]
        else:
            raise ValueError("数据集未找到有效的帧路径字段(data/img_paths)")
        
        # 提取视频目录(父目录路径)
        video_dir = os.path.dirname(frame_path)
        video_groups[video_dir].append(idx)
    
    # 转换为列表(每个元素是一个视频的所有帧索引)
    return list(video_groups.values())


def split_datasets_with_min_proportion(args, dataset_train, dataset_test, num_clients, min_proportion=0.1, seed=None):
    """按视频单元划分数据,确保训练/测试集占比严格一致 + 视频完整性 + 最小比例"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    print("\n" + "="*60)
    print("按视频单元划分数据集(训练/测试比例严格对齐)")
    print("="*60)
    print(f"最小数据比例阈值: {min_proportion:.1%}")
    
    # --------------------------
    # 1. 按视频分组(训练集+测试集)
    # --------------------------
    train_video_groups = group_indices_by_video(dataset_train)
    test_video_groups = group_indices_by_video(dataset_test)
    
    print(f"训练集共 {len(train_video_groups)} 个视频,{len(dataset_train)} 帧")
    print(f"测试集共 {len(test_video_groups)} 个视频,{len(dataset_test)} 帧")
    
    # 基础校验
    if len(test_video_groups) < num_clients:
        raise ValueError(f"测试集视频数量({len(test_video_groups)}) < 客户端数量({num_clients}),无法保证每个客户端至少1个测试视频!")
    if len(train_video_groups) < num_clients:
        raise ValueError(f"训练集视频数量({len(train_video_groups)}) < 客户端数量({num_clients}),无法保证每个客户端至少1个训练视频!")
    
    # --------------------------
    # 2. 计算视频帧数(用于比例计算)
    # --------------------------
    train_video_sizes = [len(indices) for indices in train_video_groups]
    total_train_frames = sum(train_video_sizes)
    test_video_sizes = [len(indices) for indices in test_video_groups]
    total_test_frames = sum(test_video_sizes)
    
    # --------------------------
    # 3. 分配训练集视频(确保最小比例)
    # --------------------------
    max_attempts = 100
    for attempt in range(max_attempts):
        shuffled_train_videos = list(enumerate(train_video_groups))
        random.shuffle(shuffled_train_videos)
        
        client_train_videos = [[] for _ in range(num_clients)]
        client_train_sizes = [0] * num_clients
        
        for _, frame_indices in shuffled_train_videos:
            video_size = len(frame_indices)
            min_client_idx = np.argmin(client_train_sizes)
            client_train_videos[min_client_idx].extend(frame_indices)
            client_train_sizes[min_client_idx] += video_size
        
        # 计算最终训练占比(核心:后续测试集按此占比分配)
        client_train_proportions = [s / total_train_frames for s in client_train_sizes]
        if all(prop >= min_proportion for prop in client_train_proportions):
            print(f"成功生成训练集分配 (尝试 {attempt+1} 次)")
            break
    else:
        print(f"警告: {max_attempts} 次尝试后未满足最小比例,使用均匀分配")
        shuffled_train_videos = list(enumerate(train_video_groups))
        random.shuffle(shuffled_train_videos)
        for i, (_, frame_indices) in enumerate(shuffled_train_videos):
            client_idx = i % num_clients
            client_train_videos[client_idx].extend(frame_indices)
            client_train_sizes[client_idx] += len(frame_indices)
        client_train_proportions = [s / total_train_frames for s in client_train_sizes]
    
    # --------------------------
    # 4. 分配测试集视频(核心:严格对齐训练占比)
    # --------------------------
    shuffled_test_videos = list(enumerate(test_video_groups))
    random.shuffle(shuffled_test_videos)
    client_test_videos = [[] for _ in range(num_clients)]
    client_test_sizes = [0] * num_clients
    remaining_test_videos = shuffled_test_videos.copy()
    
    # 步骤1:计算每个客户端测试集目标帧数(严格对齐训练占比)
    client_test_targets = [int(total_test_frames * prop) for prop in client_train_proportions]
    # 修正目标帧数总和(避免取整误差)
    client_test_targets[-1] = total_test_frames - sum(client_test_targets[:-1])
    # 确保每个客户端至少1个视频(目标帧数≥最小视频帧数)
    min_test_video_size = min(test_video_sizes)
    for i in range(num_clients):
        client_test_targets[i] = max(min_test_video_size, client_test_targets[i])
    
    # 步骤2:按目标帧数分配测试视频(保证视频完整性)
    for client_idx in range(num_clients):
        target = client_test_targets[client_idx]
        current_size = 0
        
        # 遍历剩余视频,直到达到目标帧数(不拆分视频)
        to_remove = []
        for idx, (_, frame_indices) in enumerate(remaining_test_videos):
            video_size = len(frame_indices)
            # 若添加该视频不超目标,或当前无视频(强制至少1个)
            if (current_size + video_size <= target) or (current_size == 0):
                client_test_videos[client_idx].extend(frame_indices)
                client_test_sizes[client_idx] += video_size
                current_size += video_size
                to_remove.append(idx)
                # 达到目标则停止
                if current_size >= target:
                    break
        
        # 移除已分配的视频
        for idx in reversed(to_remove):
            del remaining_test_videos[idx]
    
    # 步骤3:分配剩余测试视频(优先补充占比缺口最大的客户端)
    while remaining_test_videos:
        # 计算每个客户端当前测试占比 vs 训练占比的差值
        client_test_proportions = [s / total_test_frames for s in client_test_sizes]
        diffs = [abs(p - client_train_proportions[i]) for i, p in enumerate(client_test_proportions)]
        # 给差值最大的客户端分配剩余视频
        max_diff_idx = np.argmax(diffs)
        _, frame_indices = remaining_test_videos.pop(0)
        client_test_videos[max_diff_idx].extend(frame_indices)
        client_test_sizes[max_diff_idx] += len(frame_indices)
    
    # 步骤4:最终校验(确保无空客户端)
    for client_idx in range(num_clients):
        if len(client_test_videos[client_idx]) == 0:
            # 从样本最多的客户端借1个视频
            max_client_idx = np.argmax(client_test_sizes)
            borrowed_frames = client_test_videos[max_client_idx][:min_test_video_size]
            del client_test_videos[max_client_idx][:min_test_video_size]
            client_test_videos[client_idx].extend(borrowed_frames)
            client_test_sizes[client_idx] += len(borrowed_frames)
            client_test_sizes[max_client_idx] -= len(borrowed_frames)
            print(f"警告:客户端{client_idx}测试集为空,已补充1个视频")
    
    # --------------------------
    # 5. 生成客户端数据集
    # --------------------------
    client_train_datasets = [torch.utils.data.Subset(dataset_train, idx) for idx in client_train_videos]
    client_test_datasets = [torch.utils.data.Subset(dataset_test, idx) for idx in client_test_videos]
    
    # --------------------------
    # 6. 打印分配详情(重点展示比例一致性)
    # --------------------------
    print(f"\n客户端目标分配比例(训练/测试对齐):")
    for i, prop in enumerate(client_train_proportions):
        print(f"  客户端 {i}: {prop:.2%}")
    
    print("\n" + "="*60)
    print("客户端数据分配详情(训练/测试比例对齐)")
    print("="*60)
    
    total_train = 0
    total_test = 0
    for client_idx in range(num_clients):
        train_cnt = len(client_train_datasets[client_idx])
        test_cnt = len(client_test_datasets[client_idx])
        train_prop = train_cnt / len(dataset_train)
        test_prop = test_cnt / len(dataset_test)
        
        total_train += train_cnt
        total_test += test_cnt
        
        print(f"\n客户端 {client_idx}:")
        print(f"  训练样本: {train_cnt} 个 (占比: {train_prop:.2%})")
        print(f"  测试样本: {test_cnt} 个 (占比: {test_prop:.2%})")
        print(f"  比例偏差: {abs(train_prop - test_prop):.4f} (越小越好)")
    
    print("\n" + "="*60)
    print("总体统计信息")
    print("="*60)
    print(f"训练集总样本: {len(dataset_train)} | 已分配: {total_train} ({total_train/len(dataset_train):.2%})")
    print(f"测试集总样本: {len(dataset_test)} | 已分配: {total_test} ({total_test/len(dataset_test):.2%})")
    
    return client_train_datasets, client_test_datasets, client_train_proportions