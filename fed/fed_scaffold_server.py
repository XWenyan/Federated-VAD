# fed_baseline_server.py (SCAFFOLD 版本)
import torch
import numpy as np
from collections import OrderedDict


class FedMAEBaselineServer:
    """SCAFFOLD 联邦学习服务器：维护全局参数和全局控制变量"""

    def __init__(self, base_mae_model, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 从基础模型提取可训练参数（解码器参数）作为全局模型参数
        self.global_params = self._extract_trainable_params(base_mae_model)

        # 初始化全局控制变量 c（与全局参数相同形状，初始为0）
        self.global_control = {
            name: torch.zeros_like(param) for name, param in self.global_params.items()
        }

        print(f"[Fed SCAFFOLD Server] Initialized with {len(self.global_params)} parameters")
        print(f"[Fed SCAFFOLD Server] Initialized global control variables")

    def _extract_trainable_params(self, model):
        """提取需要聚合的参数（即解码器参数）"""
        trainable_params = {}
        for name, param in model.named_parameters():
            if 'decoder' in name and param.requires_grad:
                trainable_params[name] = param.data.clone().to(self.device)
        return trainable_params

    def broadcast_global_params(self):
        """广播全局模型参数和全局控制变量"""
        return (
            {k: v.clone() for k, v in self.global_params.items()},
            {k: v.clone() for k, v in self.global_control.items()}
        )

    def aggregate_updates(self, client_updates, client_weights):
        """
        聚合客户端上传的更新
        client_updates: dict {client_id: (delta_model, delta_control)}
        """
        print(f"[Fed SCAFFOLD Server] Aggregating updates from {len(client_updates)} clients")

        total_weight = sum(client_weights.values())

        # 聚合模型参数更新
        aggregated_delta_model = OrderedDict()
        # 聚合控制变量更新
        aggregated_delta_control = OrderedDict()

        # 初始化累加器
        for param_name in self.global_params.keys():
            aggregated_delta_model[param_name] = torch.zeros_like(self.global_params[param_name])
            aggregated_delta_control[param_name] = torch.zeros_like(self.global_control[param_name])

        # 加权累加
        for client_id, (delta_model, delta_control) in client_updates.items():
            weight = client_weights[client_id]
            for name in aggregated_delta_model:
                if name in delta_model:
                    aggregated_delta_model[name] += delta_model[name] * weight
                    aggregated_delta_control[name] += delta_control[name] * weight

        # 平均
        for name in aggregated_delta_model:
            aggregated_delta_model[name] /= total_weight
            aggregated_delta_control[name] /= total_weight

        # 更新全局模型参数
        for name in aggregated_delta_model:
            self.global_params[name] += aggregated_delta_model[name]

        # 更新全局控制变量
        for name in aggregated_delta_control:
            self.global_control[name] += aggregated_delta_control[name]

        print(f"[Fed SCAFFOLD Server] Aggregation completed.")