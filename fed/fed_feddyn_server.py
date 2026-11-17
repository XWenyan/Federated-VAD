# fed_baseline_server.py (FedDyn 版本)
import torch
from collections import OrderedDict

class FedMAEBaselineServer:
    """FedDyn 联邦学习服务器：聚合客户端上传的模型参数（与 FedAvg 相同）"""
    
    def __init__(self, base_mae_model, args):
        self.args = args
        # 初始化全局参数（提取可训练的解码器参数）
        self.global_params = self._extract_trainable_params(base_mae_model)
        print(f"[FedDyn Server] Initialized with {len(self.global_params)} parameters")
    
    def _extract_trainable_params(self, model):
        """提取需要聚合的参数（解码器参数）"""
        params = {}
        for name, param in model.named_parameters():
            if 'decoder' in name and param.requires_grad:
                params[name] = param.data.clone()
        return params
    
    def broadcast_global_params(self):
        """广播全局参数"""
        return {k: v.clone() for k, v in self.global_params.items()}
    
    def aggregate_updates(self, client_updates, client_weights):
        """加权平均聚合客户端模型参数"""
        print(f"[FedDyn Server] Aggregating from {len(client_updates)} clients")
        total_weight = sum(client_weights.values())
        aggregated = OrderedDict()
        for param_name in self.global_params.keys():
            weighted_sum = None
            for client_id, params in client_updates.items():
                if param_name in params:
                    weight = client_weights[client_id]
                    if weighted_sum is None:
                        weighted_sum = params[param_name] * weight
                    else:
                        weighted_sum += params[param_name] * weight
            if weighted_sum is not None:
                aggregated[param_name] = weighted_sum / total_weight
        # 更新全局参数
        for name, value in aggregated.items():
            if name in self.global_params:
                self.global_params[name] = value
        print(f"[FedDyn Server] Updated {len(aggregated)} parameters")