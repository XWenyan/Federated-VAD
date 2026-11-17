# fed_baseline_trainer.py (SCAFFOLD 版本)
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

        # 可训练参数（解码器参数）
        self.trainable_params = self.model.get_trainable_params()
        self.param_names = [name for name, _ in self.trainable_params]   # 用于索引

        # 优化器（只优化可训练参数）
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )

        # 初始化本地控制变量 c_i（与模型参数形状相同，全零）
        self.local_control = {
            name: torch.zeros_like(param) for name, param in zip(self.param_names, self.trainable_params)
        }

        print(f"[Client {client_id}] Fed SCAFFOLD Trainer initialized")

    def _apply_gradient_correction(self, server_control):
        """
        在反向传播后，对每个可训练参数的梯度进行修正：
        grad = grad - local_control + server_control
        """
        for idx, (name, param) in enumerate(zip(self.param_names, self.trainable_params)):
            if param.grad is not None:
                # 修正梯度
                correction = self.local_control[name] - server_control[name]   # 注意符号：g - c_i + c
                param.grad.data -= correction.to(param.device)   # 相当于加上 (c - c_i)

    def evaluate(self):
        """在客户端本地测试集上评估模型（与原始相同）"""
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

    def local_train(self, num_epochs, server_control):
        """
        SCAFFOLD 本地训练
        输入: server_control - 从服务器接收的全局控制变量 c
        输出: 平均损失
        """
        self.model.train()

        # 保存训练前的参数快照（用于计算 delta_model）
        params_before = {name: param.clone() for name, param in zip(self.param_names, self.trainable_params)}

        total_loss = 0
        total_steps = 0   # 记录总更新步数

        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_steps = 0

            for batch_idx, (samples, grad_mask, targets) in enumerate(self.data_loader):
                samples = samples.to(self.device)
                targets = targets.to(self.device)
                grad_mask = grad_mask.to(self.device)

                self.optimizer.zero_grad()

                # 前向传播
                loss, _, _ = self.model(
                    samples,
                    grad_mask=grad_mask,
                    targets=targets,
                    mask_ratio=self.args.mask_ratio
                )

                loss.backward()

                # 关键：梯度修正
                self._apply_gradient_correction(server_control)

                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1
                total_steps += 1

                if batch_idx % max(1, 5000 // self.args.batch_size) == 0:
                    current_samples = batch_idx * len(samples)
                    total_samples = len(self.data_loader.dataset)
                    print(f'Client {self.client_id} - Epoch: {epoch} '
                          f'[{current_samples}/{total_samples} '
                          f'({100. * batch_idx / len(self.data_loader):.0f}%)]\tLoss: {loss.item():.6f}')

            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            total_loss += avg_epoch_loss
            print(f'Client {self.client_id} - Epoch: {epoch}, Average Loss: {avg_epoch_loss:.4f}')

        avg_loss = total_loss / num_epochs if num_epochs > 0 else 0

        # 训练结束，计算模型参数变化量 delta_model
        params_after = {name: param.clone() for name, param in zip(self.param_names, self.trainable_params)}
        delta_model = {}
        for name in params_before:
            delta_model[name] = params_after[name] - params_before[name]

        # 计算客户端控制变量的更新量：根据 SCAFFOLD 公式
        # c_i_new = c_i - c + (x_new - x_old) / (lr * K)
        # 其中 K = total_steps（本地总更新步数），lr = args.lr
        lr = self.args.lr
        K = total_steps
        if K == 0:
            # 防止除零，通常不会发生
            K = 1

        c_i_new = {}
        for name in self.local_control:
            c_i_new[name] = (self.local_control[name] - server_control[name] +
                             delta_model[name] / (lr * K))

        # 计算客户端上传的 delta_control = c_i_new - c
        delta_control = {}
        for name in c_i_new:
            delta_control[name] = c_i_new[name] - server_control[name]

        # 更新本地控制变量
        self.local_control = c_i_new

        # 将计算出的 delta_model 和 delta_control 存储，供 get_global_updates 返回
        self._delta_model = delta_model
        self._delta_control = delta_control

        return avg_loss

    def get_global_updates(self):
        """返回 (delta_model, delta_control) 供服务器聚合"""
        if hasattr(self, '_delta_model'):
            return self._delta_model, self._delta_control
        else:
            return None, None

    def get_local_state(self):
        """保存本地状态（模型参数、优化器、本地控制变量）"""
        return {
            'local_params': {name: param.clone() for name, param in zip(self.param_names, self.trainable_params)},
            'optimizer': self.optimizer.state_dict(),
            'local_control': {name: c.clone() for name, c in self.local_control.items()}
        }

    def load_global_params(self, global_params, server_control):
        """从服务器加载全局模型参数和全局控制变量"""
        self.model.load_global_params(global_params)
        # 注意：服务器控制变量只用于梯度修正，不需要存储到模型，只需临时使用
        # 我们这里不直接存储，因为 local_train 会接收它
        pass

    def load_local_state(self, local_state):
        """恢复本地状态（断点续训）"""
        self.model.load_local_params(local_state['local_params'])
        self.optimizer.load_state_dict(local_state['optimizer'])
        self.local_control = local_state['local_control']