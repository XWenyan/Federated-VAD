# fed_baseline_trainer.py (MOON 版本)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.metrics as metrics
from util.abnormal_utils import filt
from fed.fed_moon_mae import FedMAEBaseline

class FedMAEBaselineTrainer:
    def __init__(self, client_id, mae_model, args, data_loader, test_loader=None):
        self.client_id = client_id
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = data_loader
        self.test_loader = test_loader
        
        # 创建基础 MAE 模型（与原始相同）

        self.model = FedMAEBaseline(mae_model)
        self.model.to(self.device)
        
        # 创建全局模型的副本（用于对比损失中的全局表征）
        self.global_model = FedMAEBaseline(mae_model)
        self.global_model.to(self.device)
        self.global_model.eval()  # 全局模型不参与梯度更新
        
        # 上一轮本地模型的副本（用于对比损失中的负样本）
        self.prev_local_model = None   # 将在第一轮训练前设置
        
        # 优化器（只优化可训练参数）
        trainable_params = self.model.get_trainable_params()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )
        
        # 投影头：将表征映射到对比损失空间（可选，但论文推荐使用）
        self.projection_head = nn.Sequential(
            nn.Linear(args.repr_dim, args.repr_dim),
            nn.ReLU(),
            nn.Linear(args.repr_dim, args.proj_dim)
        ).to(self.device)
        # 投影头参数也加入优化器
        self.optimizer.add_param_group({'params': self.projection_head.parameters(), 'lr': args.lr})
        
        print(f"[Client {client_id}] Fed MOON Trainer initialized")

    def _get_repr(self, model, samples, grad_mask):
        """
        获取模型（FedMAEBaseline）在输入上的特征向量（投影头输出）
        """
        with torch.set_grad_enabled(model.training):
            # 使用 encode 方法（不经过 decoder 和 loss）
            features = model.encode(samples, grad_mask, self.args.mask_ratio)  # [B, D]
        proj = self.projection_head(features)
        return F.normalize(proj, dim=1)
    
    def compute_contrastive_loss(self, z_cur, z_glo, z_prev, temperature=0.5):
        """
        计算模型对比损失
        z_cur: 当前模型表征
        z_glo: 全局模型表征
        z_prev: 上一轮本地模型表征
        """
        # 相似度使用余弦相似度（已归一化，点积即余弦）
        sim_cur_glo = torch.sum(z_cur * z_glo, dim=1)   # [B]
        sim_cur_prev = torch.sum(z_cur * z_prev, dim=1) # [B]
        
        # 对比损失公式：-log( exp(sim_cur_glo/τ) / (exp(sim_cur_glo/τ) + exp(sim_cur_prev/τ)) )
        tau = temperature
        exp_glo = torch.exp(sim_cur_glo / tau)
        exp_prev = torch.exp(sim_cur_prev / tau)
        contrast_loss = -torch.log(exp_glo / (exp_glo + exp_prev + 1e-8))  # 防止除零
        return contrast_loss.mean()
    
    def local_train(self, num_epochs):
        """本地训练，加入 MOON 对比损失"""
        self.model.train()
        self.projection_head.train()
        
        # 确保全局模型和上一轮本地模型可用
        # 全局模型已经通过 load_global_params 同步，保持不变
        # 上一轮本地模型如果为 None，则用当前模型初始化（第一轮）
        if self.prev_local_model is None:
            self.prev_local_model = FedMAEBaseline(self.model.mae_model)  # 复制结构
            self.prev_local_model.load_state_dict(self.model.state_dict())
            self.prev_local_model.to(self.device)
            self.prev_local_model.eval()
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_batches = 0
            
            for batch_idx, (samples, grad_mask, targets) in enumerate(self.data_loader):
                samples = samples.to(self.device)
                targets = targets.to(self.device)
                grad_mask = grad_mask.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 1. 原始 MAE 重建损失
                loss_recon, _, _ = self.model(
                    samples,
                    grad_mask=grad_mask,
                    targets=targets,
                    mask_ratio=self.args.mask_ratio
                )
                
                # 2. 计算对比损失所需的表征向量
                # 当前模型表征
                z_cur = self._get_repr(self.model, samples, grad_mask)
                # 全局模型表征（不需要梯度）
                with torch.no_grad():
                    z_glo = self._get_repr(self.global_model, samples, grad_mask)
                    z_prev = self._get_repr(self.prev_local_model, samples, grad_mask)
                
                # 3. 对比损失
                loss_contrast = self.compute_contrastive_loss(
                    z_cur, z_glo, z_prev,
                    temperature=getattr(self.args, 'moon_temperature', 0.5)
                )
                
                # 4. 总损失
                mu = getattr(self.args, 'moon_mu', 1.0)   # 对比损失权重
                total_loss_batch = loss_recon + mu * loss_contrast
                
                total_loss_batch.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss_batch.item()
                epoch_batches += 1
                num_batches += 1
                
                if batch_idx % max(1, 5000 // self.args.batch_size) == 0:
                    current_samples = batch_idx * len(samples)
                    total_samples = len(self.data_loader.dataset)
                    print(f'Client {self.client_id} - Epoch {epoch} [{current_samples}/{total_samples} ({100.*batch_idx/len(self.data_loader):.0f}%)] '
                          f'Loss: {loss_recon.item():.6f}, Contrast: {loss_contrast.item():.6f}')
            
            epoch_avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            total_loss += epoch_avg_loss
            print(f'Client {self.client_id} - Epoch {epoch} Average Loss: {epoch_avg_loss:.4f}')
        
        avg_loss = total_loss / num_epochs if num_epochs > 0 else 0
        
        # 训练结束后，将当前模型保存为下一轮的 prev_local_model
        # 注意：深拷贝状态
        new_prev_model = FedMAEBaseline(self.model.mae_model)
        new_prev_model.load_state_dict(self.model.state_dict())
        new_prev_model.to(self.device)
        new_prev_model.eval()
        self.prev_local_model = new_prev_model
        
        return avg_loss
    
    def get_global_updates(self):
        """返回需要上传到服务器的参数更新（即模型的解码器参数）"""
        return self.model.get_global_params()
    
    def get_local_state(self):
        """保存本地状态（模型参数、优化器、prev_local_model、projection_head）"""
        return {
            'local_params': self.model.get_local_params(),
            'optimizer': self.optimizer.state_dict(),
            'prev_local_model_state': self.prev_local_model.state_dict() if self.prev_local_model else None,
            'projection_head_state': self.projection_head.state_dict()
        }
    
    def load_global_params(self, global_params):
        """加载服务器下发的全局参数，并同步到 global_model"""
        self.model.load_global_params(global_params)
        # 同时更新全局模型副本
        self.global_model.load_global_params(global_params)
        # 注意：全局模型应保持 eval 模式且不参与梯度
        self.global_model.eval()
    
    def load_local_state(self, local_state):
        """恢复本地状态（断点续训）"""
        self.model.load_local_params(local_state['local_params'])
        self.optimizer.load_state_dict(local_state['optimizer'])
        if local_state['prev_local_model_state'] is not None:
            if self.prev_local_model is None:
                self.prev_local_model = FedMAEBaseline(self.model.mae_model)
                self.prev_local_model.to(self.device)
            self.prev_local_model.load_state_dict(local_state['prev_local_model_state'])
            self.prev_local_model.eval()
        self.projection_head.load_state_dict(local_state['projection_head_state'])
    
    # evaluate 方法与原始完全相同（不变）
    def evaluate(self):
        """在客户端本地测试集上评估模型（与原始一样）"""
        if self.test_loader is None:
            return 0.0, 0.0
        self.model.eval()
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
                _, _, _, recon_error = self.model(
                    samples,
                    targets=targets,
                    grad_mask=grads,
                    mask_ratio=self.args.mask_ratio
                )
                if isinstance(recon_error, list):
                    recon_error = recon_error[0] + recon_error[1]
                recon_error = recon_error.detach().cpu().numpy()
                predictions += list(recon_error)
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
                pred = filt(pred, range=700, mu=282)
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
        filtered_preds = np.concatenate(filtered_preds)
        filtered_labels = np.concatenate(filtered_labels)
        fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
        micro_auc = metrics.auc(fpr, tpr)
        micro_auc = np.nan_to_num(micro_auc, nan=1.0)
        print(f"Client {self.client_id} - MicroAUC: {micro_auc:.4f}, MacroAUC: {macro_auc:.4f}")
        return micro_auc, macro_auc