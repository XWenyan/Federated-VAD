# fed_baseline_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
from util.abnormal_utils import filt


class FedMAEBaselineTrainer:
    def __init__(self, client_id, mae_model, args, data_loader, test_loader=None):
        self.client_id = client_id
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = data_loader
        self.test_loader = test_loader
        
        # 创建基础联邦学习MAE模型
        from fed.fed_baseline_mae import FedMAEBaseline
        self.model = FedMAEBaseline(mae_model)
        self.model.to(self.device)

        # fedprox改动
        # 用于存储服务器下发的全局模型参数 (w_t)
        self.global_model_cache = {} 
    
        # 优化器只优化本地可训练参数
        trainable_params = self.model.get_trainable_params()
        print(f"[Client {client_id}] Trainable parameters: {len(trainable_params)}")
        
        # 使用AdamW优化器
        self.optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=args.lr, 
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )
        
        print(f"[Client {client_id}] Fed Baseline Trainer initialized")

    def evaluate(self):
        """在客户端本地测试集上评估模型"""
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
                pred = filt(pred, range=700, mu=282) # 本来是900
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
        total_loss = 0
        num_batches = 0

        # fedprox修改
        # 获取mu超参数
        # 从 args 中获取 mu，如果没有设置默认为 0.0 (退化为 FedAvg)
        mu = getattr(self.args, 'mu', 0.0)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_batches = 0
            
            for batch_idx, (samples, grad_mask, targets) in enumerate(self.data_loader):                    
                samples = samples.to(self.device)
                targets = targets.to(self.device)
                grad_mask = grad_mask.to(self.device)
                
                # 优化器清零
                self.optimizer.zero_grad()

                # MAE前向传播
                loss, _, _ = self.model(
                    samples, 
                    grad_mask=grad_mask, 
                    targets=targets, 
                    mask_ratio=self.args.mask_ratio
                )
                 # fedprox修改 添加fedprox近端项
                proximal_term = 0.0
                if mu > 0.0:
                    # 计算 ||w - w_global||^2
                    # 只计算解码器参数（因为编码器不参与联邦聚合，理论上不需要约束，
                    # 但为了代码通用性，这里只遍历 global_model_cache 中存在的参数）
                    for name, param in self.model.mae_model.named_parameters():
                        if name in self.global_model_cache and param.requires_grad:
                            # 获取全局模型对应的参数
                            global_param = self.global_model_cache[name].to(self.device)
                            # 累加 L2 范数平方
                            proximal_term += (param - global_param).norm(2) ** 2
                    # 总损失 = 原始损失 + (mu/2) * 近端项
                    loss_fedprox = loss + (mu / 2) * proximal_term
                    loss_fedprox.backward()
                else:
                    # 如果 mu=0，则退化为 FedAvg
                    loss.backward()

                self.optimizer.step()
               
                epoch_loss += loss_fedprox.item() # 记录总Loss
                epoch_batches += 1
                
                # 每5000个样本打印一次
                if batch_idx % max(1, 5000 // self.args.batch_size) == 0:
                    current_samples = batch_idx * len(samples)
                    total_samples = len(self.data_loader.dataset)
                    print(f'Client {self.client_id} - Epoch: {epoch} '
                          f'[{current_samples}/{total_samples} '
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
        }

    def load_global_params(self, global_params):
        """从服务器加载全局参数"""
        # fedprox修改 更新缓存
        self.global_model_cache = {k: v.clone().cpu() for k, v in global_params.items()}
        self.model.load_global_params(global_params)

    def load_local_state(self, local_state):
        """加载本地状态"""
        self.model.load_local_params(local_state['local_params'])
        self.optimizer.load_state_dict(local_state['optimizer'])