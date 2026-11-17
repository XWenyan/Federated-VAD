import torch
import copy
from collections import OrderedDict
import os

class FedMAEServer:
    def __init__(self, global_mae_model, args):
        self.args = args
        
        # 创建临时联邦学习模型来获取初始全局参数
        from fed.prefix_mae_ge import FedMAEPrefix
        temp_model = FedMAEPrefix(
            global_mae_model, 
            mid_dim=args.fed_mid_dim,
            scale=args.fed_scale,
            local_parts=args.fed_local_parts,
            vanilla=args.fed_vanilla,
            depth=args.fed_prefix_depth
        )
        
        # 初始化全局参数
        self.global_params = temp_model.get_global_params()
        
        print(f"Server initialized with {len(self.global_params)} global parameters")

    # 添加属性访问方法
    @property
    def global_params(self):
        """获取当前全局参数"""
        return self._global_params
    
    @global_params.setter
    def global_params(self, value):
        """设置全局参数"""
        self._global_params = value

    def aggregate_updates(self, client_updates, client_weights=None):
        """聚合客户端更新（加权平均）"""
        print("Aggregating client updates...")
        
        # 如果没有指定权重，使用均匀权重
        if client_weights is None:
            client_weights = {client_id: 1.0 for client_id in client_updates.keys()}
        
        # 归一化权重
        total_weight = sum(client_weights.values())
        normalized_weights = {k: v/total_weight for k, v in client_weights.items()}
        
        # 初始化聚合参数
        aggregated_params = OrderedDict()
        for client_id, updates in client_updates.items():
            weight = normalized_weights[client_id]
            for key, param in updates.items():
                if key not in aggregated_params:
                    aggregated_params[key] = param * weight
                else:
                    aggregated_params[key] += param * weight
        
        # 更新全局参数
        for key in self.global_params.keys():
            if key in aggregated_params:
                self.global_params[key] = aggregated_params[key]
        
        print(f"Aggregation completed. Updated {len(aggregated_params)} parameters.")
        return self.global_params

    def broadcast_global_params(self):
        """广播全局参数给所有客户端"""
        return copy.deepcopy(self._global_params)

    def save_global_model(self, path):
        """保存全局模型"""
        torch.save({
            'global_params': self.global_params,
            'args': self.args
        }, path)
        print(f"Global model saved to {path}")

    def load_global_model(self, path):
        """加载全局模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.global_params = checkpoint['global_params']
            print(f"Global model loaded from {path}")
        else:
            print(f"Checkpoint {path} not found, using initial global parameters")