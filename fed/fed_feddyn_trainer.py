# fed_baseline_trainer.py (FedDyn 版本)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
from copy import deepcopy
from util.abnormal_utils import filt


class FedMAEBaselineTrainer:
    def __init__(self, client_id, mae_model, args, data_loader, test_loader=None):
        self.client_id = client_id
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = data_loader
        self.test_loader = test_loader
        
        # 创建本地模型（包装 MAE）
        from fed.fed_baseline_mae import FedMAEBaseline
        self.model = FedMAEBaseline(mae_model)
        self.model.to(self.device)
        
        # 优化器（只优化可训练参数）
        trainable_params = self.model.get_trainable_params()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )
        
        # FedDyn 专用：初始化 h_i 为全零（与可训练参数形状相同）
        self.h_i = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.h_i[name] = torch.zeros_like(param.data)
        
        # 保存全局模型副本（用于计算惩罚项）
        self.global_model = FedMAEBaseline(mae_model)
        self.global_model.to(self.device)
        self.global_model.eval()
        
        # FedDyn 超参数
        self.alpha = getattr(args, 'feddyn_alpha', 0.01)   # 惩罚系数
        print(f"[Client {client_id}] FedDyn initialized, alpha={self.alpha}")
    
    def compute_feddyn_loss(self, loss_original):
        """
        在原始损失基础上添加 FedDyn 正则项：
        L_total = L_original + (α/2)*||w - w_global||^2 - ⟨h_i, w⟩
        """
        feddyn_reg = 0.0
        linear_term = 0.0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # 获取全局模型中对应参数
            global_param = self.global_model.state_dict().get(name)
            if global_param is None:
                continue
            # (α/2) * ||w - w_global||^2
            diff = param - global_param.to(param.device)
            feddyn_reg += 0.5 * self.alpha * torch.sum(diff ** 2)
            # - ⟨h_i, w⟩
            h = self.h_i[name].to(param.device)
            linear_term -= torch.sum(h * param)
        total_loss = loss_original + feddyn_reg + linear_term
        return total_loss
    
    def local_train(self, num_epochs):
        """FedDyn 本地训练：使用修正后的损失"""
        self.model.train()
        
        # 加载当前全局模型参数到 global_model（确保一致）
        # 注意：global_model 应该在每轮训练开始时由外部调用 load_global_params 同步
        # 这里假设在调用 local_train 之前已经调用过 load_global_params
        
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
                
                # 原始 MAE 损失
                loss_original, _, _ = self.model(
                    samples,
                    grad_mask=grad_mask,
                    targets=targets,
                    mask_ratio=self.args.mask_ratio
                )
                
                # 加上 FedDyn 正则项
                total_loss_batch = self.compute_feddyn_loss(loss_original)
                
                total_loss_batch.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss_batch.item()
                epoch_batches += 1
                num_batches += 1
                
                if batch_idx % max(1, 5000 // self.args.batch_size) == 0:
                    current_samples = batch_idx * len(samples)
                    total_samples = len(self.data_loader.dataset)
                    print(f'Client {self.client_id} - Epoch {epoch} [{current_samples}/{total_samples} ({100.*batch_idx/len(self.data_loader):.0f}%)] '
                          f'Loss: {loss_original.item():.6f}, FedDynTotal: {total_loss_batch.item():.6f}')
            
            epoch_avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            total_loss += epoch_avg_loss
            print(f'Client {self.client_id} - Epoch {epoch} Average Loss: {epoch_avg_loss:.4f}')
        
        avg_loss = total_loss / num_epochs if num_epochs > 0 else 0
        
        # 训练结束后，更新 h_i： h_i = h_i - α * (w - w_global)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.h_i:
                    global_param = self.global_model.state_dict().get(name)
                    if global_param is not None:
                        diff = param - global_param.to(param.device)
                        self.h_i[name] = self.h_i[name] - self.alpha * diff
        
        return avg_loss
    
    def get_global_updates(self):
        """返回当前的模型参数（用于上传到服务器）"""
        return self.model.get_global_params()
    
    def get_local_state(self):
        """保存本地状态（模型参数、优化器、h_i）"""
        return {
            'local_params': self.model.get_local_params(),
            'optimizer': self.optimizer.state_dict(),
            'h_i': {k: v.cpu().clone() for k, v in self.h_i.items()}
        }
    
    def load_global_params(self, global_params):
        """加载服务器下发的全局参数，并同步到 model 和 global_model"""
        self.model.load_global_params(global_params)
        # 同时更新全局模型副本
        self.global_model.load_global_params(global_params)
        # 确保 global_model 处于 eval 模式
        self.global_model.eval()
    
    def load_local_state(self, local_state):
        """恢复本地状态"""
        self.model.load_local_params(local_state['local_params'])
        self.optimizer.load_state_dict(local_state['optimizer'])
        for name, h_val in local_state['h_i'].items():
            if name in self.h_i:
                self.h_i[name] = h_val.to(self.device)
    
    def evaluate(self):
        """评估函数（与原始一致，略）"""
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