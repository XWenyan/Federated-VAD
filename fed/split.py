import random
import numpy as np
import torch
from collections import defaultdict

class FederatedDataSplitter:
    def __init__(self, dataset, client_num, random_seed=42):
        """
        联邦学习数据分配器        
        Args:
            dataset: AbnormalDatasetGradientsTrain 实例
            client_num: 客户端数量
            random_seed: 随机种子
        """
        self.dataset = dataset
        self.client_num = client_num
        self.random_seed = random_seed
        self.client_data_indices = None
        
    def split_data(self, split_method='iid', **kwargs):
        """
        分割数据到各个客户端  
        Args:
            split_method: 分割方法 ('iid', 'non_iid')
            **kwargs: 其他参数,如non_iid时的类别分布参数
        """
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        total_samples = len(self.dataset)
        indices = list(range(total_samples))
        
        if split_method == 'iid':
            self._split_iid(indices)
        elif split_method == 'non_iid':
            self._split_non_iid(indices, **kwargs)
        else:
            raise ValueError(f"不支持的split_method: {split_method}")
    
    def _split_iid(self, indices):
        """IID分割:均匀随机分配"""
        random.shuffle(indices)
        
        # 计算每个客户端的数据量
        samples_per_client = len(indices) // self.client_num
        remainder = len(indices) % self.client_num
        
        self.client_data_indices = {}
        start_idx = 0
        
        for client_id in range(self.client_num):
            # 分配数据，处理余数
            end_idx = start_idx + samples_per_client + (1 if client_id < remainder else 0)
            self.client_data_indices[client_id] = indices[start_idx:end_idx]
            start_idx = end_idx
    
    def _split_non_iid(self, indices, alpha=0.5):
        """
        Non-IID分割:基于狄利克雷分布 
        Args:
            alpha: 狄利克雷分布参数，越小越不均匀
        """
        # 简化实现，实际可以根据视频ID等进行更复杂的non-iid分割
        random.shuffle(indices)
        
        # 使用狄利克雷分布生成不均匀分配
        proportions = np.random.dirichlet(np.repeat(alpha, self.client_num))
        proportions = proportions / proportions.sum()
        client_sizes = (proportions * len(indices)).astype(int)
        
        # 调整大小确保总和正确
        diff = len(indices) - client_sizes.sum()
        if diff > 0:
            client_sizes[:diff] += 1
        
        self.client_data_indices = {}
        start_idx = 0
        
        for client_id in range(self.client_num):
            end_idx = start_idx + client_sizes[client_id]
            self.client_data_indices[client_id] = indices[start_idx:end_idx]
            start_idx = end_idx
    
    def get_client_dataset(self, client_id):
        # 获取指定客户端的数据集子集
        if self.client_data_indices is None:
            raise ValueError("请先调用split_data方法分割数据")
        
        if client_id not in self.client_data_indices:
            raise ValueError(f"客户端ID {client_id} 不存在")
        
        return ClientSubsetDataset(self.dataset, self.client_data_indices[client_id])
    
    def get_all_client_datasets(self):
        # 获取所有客户端的数据集
        if self.client_data_indices is None:
            raise ValueError("请先调用split_data方法分割数据")
        
        client_datasets = {}
        for client_id in range(self.client_num):
            client_datasets[client_id] = self.get_client_dataset(client_id)
        
        return client_datasets
    
    def get_client_data_info(self):
        """获取每个客户端的数据信息（用于客户端初始化）"""
        if self.client_data_indices is None:
            raise ValueError("请先调用split_data方法分割数据")
        
        client_info = {}
        for client_id in range(self.client_num):
            indices = self.client_data_indices[client_id]
            client_info[client_id] = {
                'train_samples': len(indices),
                'test_samples': 0,  # 暂时设为0，后续可以添加测试集
                'train_dataset': self.get_client_dataset(client_id)
            }
        
        return client_info
    
    def print_client_statistics(self):
        # 打印每个客户端的数据统计信息
        if self.client_data_indices is None:
            print("数据尚未分配")
            return       
        print(f"{'='*60}")   
        total_samples = len(self.dataset)
        
        for client_id in range(self.client_num):
            client_indices = self.client_data_indices[client_id]
            client_samples = len(client_indices)
            
            # 抽样检查数据维度
            if client_samples > 0:
                sample_idx = client_indices[0]
                img, gradient, target = self.dataset[sample_idx]
                
                print(f"客户端 {client_id:2d}:")
                print(f"  - 数据量: {client_samples:5d} 样本 "
                      f"({client_samples/total_samples*100:.1f}%)")
                print(f"  - 输入图像维度: {img.shape}")
                print(f"  - 梯度图像维度: {gradient.shape}")
                print(f"  - 目标图像维度: {target.shape}")
                
                # 统计3D样本数量（如果适用）
                if hasattr(self.dataset, 'input_3d') and self.dataset.input_3d:
                    print(f"  - 3D输入: 是 (通道数: {img.shape[0]})")
                else:
                    print(f"  - 3D输入: 否")
            else:
                print(f"客户端 {client_id:2d}: 无数据")
            
            print(f"{'-'*40}")
        
        # 总体统计
        print(f"\n总体统计:")
        print(f"总样本数: {total_samples}")
        print(f"平均每个客户端样本数: {total_samples/self.client_num:.1f}")
        sample_counts = [len(indices) for indices in self.client_data_indices.values()]
        print(f"样本数标准差: {np.std(sample_counts):.1f}")
        print(f"最小样本数: {min(sample_counts)}")
        print(f"最大样本数: {max(sample_counts)}")
        print(f"{'='*60}")


class ClientSubsetDataset(torch.utils.data.Dataset):
    # 客户端数据集子集 
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    
    def __len__(self):
        return len(self.indices)
    
    def __repr__(self):
        return f"ClientSubsetDataset(size={len(self)})"