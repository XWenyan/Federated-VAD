# fed_moon_mae.py
import torch
import torch.nn as nn
from copy import deepcopy


class FedMAEBaseline(nn.Module):
    """基础联邦学习MAE:编码器本地化,解码器参与联邦通信（无前缀适配器）"""
    
    def __init__(self, mae_model):
        super(FedMAEBaseline, self).__init__()
        
        # 深拷贝MAE模型
        self.mae_model = deepcopy(mae_model)
        
        # 设置参数冻结策略
        self._setup_parameter_groups()

    def _setup_parameter_groups(self):
        """设置参数组：编码器本地化，解码器参与联邦通信"""
        # 首先冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 解冻编码器参数（本地）
        for name, param in self.mae_model.named_parameters():
            if 'encoder' in name or 'blocks' in name or 'patch_embed' in name or 'cls_token' in name:
                param.requires_grad = True
                
        # 解冻解码器参数（参与联邦通信）
        for name, param in self.mae_model.named_parameters():
            if 'decoder' in name:
                param.requires_grad = True

    def get_global_params(self):
        """获取全局参数：解码器所有参数"""
        global_params = {}
        
        # 解码器所有参数
        for name, param in self.mae_model.named_parameters():
            if 'decoder' in name and param.requires_grad:
                global_params[name] = param.data.clone()
        
        print(f"[Fed Baseline] Collected {len(global_params)} global parameters from decoder")
        return global_params

    def get_local_params(self):
        """获取本地参数：编码器所有参数"""
        local_params = {}
        
        # 编码器所有参数
        for name, param in self.mae_model.named_parameters():
            if ('encoder' in name or 'blocks' in name or 'patch_embed' in name or 
                'cls_token' in name) and param.requires_grad:
                local_params[name] = param.data.clone()
        
        print(f"[Fed Baseline] Collected {len(local_params)} local parameters from encoder")
        return local_params

    def load_global_params(self, global_params):
        """加载全局参数到解码器"""
        loaded_count = 0
        for key, global_param in global_params.items():
            try:
                # 使用递归获取模块
                keys = key.split('.')
                obj = self.mae_model
                for k in keys[:-1]:
                    obj = getattr(obj, k)
                
                # 更新参数
                param_name = keys[-1]
                if hasattr(obj, param_name):
                    param = getattr(obj, param_name)
                    param.data.copy_(global_param)
                    loaded_count += 1
                else:
                    print(f"Warning: {key} not found in model")
                    
            except (AttributeError, KeyError) as e:
                print(f"Error loading global parameter {key}: {e}")
                continue
        
        print(f"[Fed Baseline] Loaded {loaded_count} global parameters to decoder")

    def load_local_params(self, local_params):
        """加载本地参数到编码器"""
        loaded_count = 0
        for key, local_param in local_params.items():
            try:
                # 使用递归获取模块
                keys = key.split('.')
                obj = self.mae_model
                for k in keys[:-1]:
                    obj = getattr(obj, k)
                
                # 更新参数
                param_name = keys[-1]
                if hasattr(obj, param_name):
                    param = getattr(obj, param_name)
                    param.data.copy_(local_param)
                    loaded_count += 1
                else:
                    print(f"Warning: {key} not found in model")
                    
            except (AttributeError, KeyError) as e:
                print(f"Error loading local parameter {key}: {e}")
                continue
        
        print(f"[Fed Baseline] Loaded {loaded_count} local parameters to encoder")


    def encode(self, imgs, grad_mask=None, mask_ratio=0.75):
        """
        仅运行编码器，返回全局特征向量（取 cls_token）
        """
        # 注意参数顺序：先 mask_ratio，再 grad_mask
        latent, _, _ = self.mae_model.forward_encoder(imgs, mask_ratio, grad_mask)
        cls_feat = latent[:, 0, :]   # [B, D]
        return cls_feat   
    
    def forward(self, imgs, targets=None, grad_mask=None, mask_ratio=0.75, scene_embed=None):
        """MAE前向传播"""
        # 忽略scene_embed参数，调用原始MAE
        return self.mae_model(imgs, targets, grad_mask, mask_ratio)

    def get_trainable_params(self):
        """获取当前需要训练的参数（用于优化器）"""
        return [param for param in self.parameters() if param.requires_grad]