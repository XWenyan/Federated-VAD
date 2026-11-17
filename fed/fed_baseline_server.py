# fed_baseline_server.py
import torch
import numpy as np
from collections import OrderedDict


class FedMAEBaselineServer:
    """基础联邦学习服务器：聚合客户端上传的解码器参数"""
    
    def __init__(self, base_mae_model, args):
        self.args = args
        
        # 初始化全局参数（从基础模型获取解码器参数）
        self.global_params = self._extract_decoder_params(base_mae_model)
        
        print(f"[Fed Baseline Server] Initialized with {len(self.global_params)} global parameters")
        
    def _extract_decoder_params(self, model):
        """从基础模型提取解码器参数作为初始全局参数"""
        decoder_params = {}
        
        for name, param in model.named_parameters():
            if 'decoder' in name and param.requires_grad:
                decoder_params[name] = param.data.clone()
        
        return decoder_params
    
    def broadcast_global_params(self):
        """广播全局参数给所有客户端"""
        return {k: v.clone() for k, v in self.global_params.items()}
    
    def aggregate_updates(self, client_updates, client_weights):
        """聚合客户端上传的参数更新"""
        print(f"[Fed Baseline Server] Aggregating updates from {len(client_updates)} clients")
        
        # 计算总权重
        total_weight = sum(client_weights.values())
        
        # 初始化聚合后的参数
        aggregated_params = {}
        
        # 对每个参数进行加权平均
        for param_name in self.global_params.keys():
            weighted_sum = None
            
            for client_id, client_update in client_updates.items():
                if param_name in client_update:
                    client_weight = client_weights[client_id]
                    
                    if weighted_sum is None:
                        weighted_sum = client_update[param_name] * client_weight
                    else:
                        weighted_sum += client_update[param_name] * client_weight
            
            if weighted_sum is not None:
                aggregated_params[param_name] = weighted_sum / total_weight
        
        # 更新全局参数
        for param_name, param_value in aggregated_params.items():
            if param_name in self.global_params:
                self.global_params[param_name] = param_value
        
        print(f"[Fed Baseline Server] Aggregation completed. Updated {len(aggregated_params)} parameters")