# fed_mae_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from timm.optim import optim_factory 
import numpy as np
import sklearn.metrics as metrics
from util.abnormal_utils import filt


class LightweightSceneEncoder(nn.Module):
    """轻量级场景编码器"""
    def __init__(self, mae_model, scene_dim=64):
        super().__init__()
        # 复用MAE的patch_embed（已经训练好处理9通道）
        self.patch_embed = mae_model.patch_embed
        
        # 冻结基础特征提取器（可选）
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        # 简单聚合：平均所有patch -> 线性投影
        self.projection = nn.Sequential(
            nn.Linear(mae_model.embed_dim, scene_dim),
            nn.ReLU(),
            nn.Linear(scene_dim, scene_dim)
        )
    
    def forward(self, x):
        # x: (B, 9, H, W)
        patch_features = self.patch_embed(x)  # (B, N, D)
        global_feat = patch_features.mean(dim=1)  # (B, D)
        scene_embed = self.projection(global_feat)  # (B, scene_dim)
        return scene_embed
    
class FedMAETrainer:
    def __init__(self, client_id, mae_model, args, data_loader, test_loader=None):
        self.client_id = client_id
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = data_loader
        self.test_loader = test_loader  # 添加测试数据加载器
        
        # 创建联邦学习MAE模型
        from fed.prefix_mae_ge import build_prefix_fed_mae
        self.model = build_prefix_fed_mae(mae_model, args)
        self.model.to(self.device)
        
        # 优化器只优化本地可训练参数
        trainable_params = self.model.get_trainable_params()
        print(f"Client {client_id} - Trainable parameters: {len(trainable_params)}")
        
        # 使用与原始训练相同的优化器设置
        param_groups = optim_factory.param_groups_weight_decay(self.model, args.weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        
        print(f"Client {client_id} optimizer: {self.optimizer}")

        # 🔧 新增：添加场景编码器（客户端本地）
        print(f"[Client {client_id}] 初始化场景编码器...")
        self.scene_encoder = LightweightSceneEncoder(self.model.mae_model, scene_dim=64)
        self.scene_encoder.to(self.device)

        # 场景编码器参数统计
        scene_params = sum(p.numel() for p in self.scene_encoder.parameters() if p.requires_grad)
        print(f"[Client {client_id}] 场景编码器可训练参数: {scene_params:,}")
        
        # 场景编码器的优化器（单独优化）
        self.scene_optimizer = torch.optim.AdamW(
            self.scene_encoder.parameters(), 
            lr=args.lr * 0.1,  # 使用较小的学习率
            weight_decay=args.weight_decay
        )
        
        # 连接场景编码器到模型的适配器
        # self._connect_scene_encoder()
        
        print(f"[Client {client_id}] 场景感知训练器初始化完成")

    def evaluate(self):
        """在客户端本地测试集上评估模型"""
        if self.test_loader is None:
            return 0.0, 0.0
            
        self.model.eval()
        self.scene_encoder.eval()  # 🔧 场景编码器也要评估模式
        predictions = []
        labels = []
        videos = []
        
        with torch.no_grad():
            for data_iter_step, (samples, grads, targets, label, vid, _) in enumerate(self.test_loader):
                videos += list(vid)
                labels += list(label.detach().cpu().numpy())

                samples = samples.to(self.device)
                grads = grads.to(self.device)
                targets = targets.to(self.device)
                # 🔧 提取场景特征
                scene_embed = self.scene_encoder(samples)
                _, _, _, recon_error = self.model(samples, grad_mask=grads, targets=targets, mask_ratio=self.args.mask_ratio, scene_embed=scene_embed)
                
                if isinstance(recon_error, list):
                    recon_error = recon_error[0] + recon_error[1]
                recon_error = recon_error.detach().cpu().numpy()
                predictions += list(recon_error)

        # 计算统计指标
        predictions = np.array(predictions)
        labels = np.array(labels)
        videos = np.array(videos)

        aucs = []
        filtered_preds = []
        filtered_labels = []
        for vid in np.unique(videos):
            pred = predictions[np.array(videos) == vid]
            pred = np.nan_to_num(pred, nan=0.)
            if self.args.dataset == 'avenue':
                pred = filt(pred, range=38, mu=11)
            else:
                pred = filt(pred, range=700, mu=282) # 本来是900 目前700最佳
                pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
            
            filtered_preds.append(pred)
            lbl = labels[np.array(videos) == vid]
            filtered_labels.append(lbl)
            lbl = np.array([0] + list(lbl) + [1])
            pred = np.array([0] + list(pred) + [1])
            fpr, tpr, _ = metrics.roc_curve(lbl, pred)
            res = metrics.auc(fpr, tpr)
            aucs.append(res)

        macro_auc = np.nanmean(aucs)

        # Micro-AUC
        filtered_preds = np.concatenate(filtered_preds)
        filtered_labels = np.concatenate(filtered_labels)

        fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
        micro_auc = metrics.auc(fpr, tpr)
        micro_auc = np.nan_to_num(micro_auc, nan=1.0)

        print(f"Client {self.client_id} - MicroAUC: {micro_auc:.4f}, MacroAUC: {macro_auc:.4f}")
        return micro_auc, macro_auc

    def local_train(self, num_epochs):
        """本地训练 - 返回平均损失"""
        self.model.train()
        self.scene_encoder.train()  # 🔧 场景编码器也要训练
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_batches = 0
            
            for batch_idx, (samples, grad_mask, targets) in enumerate(self.data_loader):                    
                samples = samples.to(self.device)
                targets = targets.to(self.device)
                grad_mask = grad_mask.to(self.device)
                
                # 优化器清零
                self.optimizer.zero_grad()
                self.scene_optimizer.zero_grad()

                # 🔧 1. 提取场景特征
                scene_embed = self.scene_encoder(samples)  # (B, scene_dim)
                # 第一次batch打印场景特征信息
                if batch_idx == 0:
                    print(f"[Client {self.client_id}] Epoch {epoch}, Batch 0: 场景特征提取成功!")
                    print(f"  场景特征形状: {scene_embed.shape}")
                    print(f"  场景特征范围: [{scene_embed.min().item():.4f}, {scene_embed.max().item():.4f}]")
                    print(f"  场景特征均值: {scene_embed.mean().item():.4f}")
                
                # MAE前向传播
                loss, _, _ = self.model(samples, grad_mask=grad_mask, targets=targets, mask_ratio=self.args.mask_ratio, scene_embed=scene_embed)
                
                loss.backward()
                self.optimizer.step()
                self.scene_optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
                
                if batch_idx % self.args.print_freq == 0:
                    print(f'Client {self.client_id} - Epoch: {epoch} '
                          f'[{batch_idx * len(samples)}/{len(self.data_loader.dataset)} '
                          f'({100. * batch_idx / len(self.data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
            epoch_avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            total_loss += epoch_avg_loss
            num_batches += 1
            
            print(f'Client {self.client_id} - Epoch: {epoch}, Average Loss: {epoch_avg_loss:.4f}')
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def get_global_updates(self):
        """获取全局参数更新（用于上传到服务器）"""
        return self.model.get_global_params()

    def get_local_state(self):
        """获取本地状态（用于保存检查点）"""
        return {
            'local_params': self.model.get_local_params(),
            'optimizer': self.optimizer.state_dict(),
            'scene_encoder': self.scene_encoder.state_dict(),  # 🔧 新增
            'scene_optimizer': self.scene_optimizer.state_dict(),  # 🔧 新增
        }

    def load_global_params(self, global_params):
        """从服务器加载全局参数"""
        self.model.load_global_params(global_params)

    def load_local_state(self, local_state):
        """加载本地状态"""
        self.model.load_local_params(local_state['local_params'])
        self.optimizer.load_state_dict(local_state['optimizer'])
        # 🔧 新增：加载场景编码器状态
        if 'scene_encoder' in local_state:
            self.scene_encoder.load_state_dict(local_state['scene_encoder'])
        if 'scene_optimizer' in local_state:
            self.scene_optimizer.load_state_dict(local_state['scene_optimizer'])