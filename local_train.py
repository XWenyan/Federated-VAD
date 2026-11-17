# local_mae_train.py
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import ml_collections
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.local_configs import get_configs_avenue, get_configs_shanghai
from data.test_dataset import AbnormalDatasetGradientsTest
from data.train_dataset import AbnormalDatasetGradientsTrain
from model.local_mae import LocalMaskedAutoencoderViT
from util.abnormal_utils import filt


class LocalMAETrainer:
    """完全本地化的MAE训练器"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建数据集
        print("Loading datasets...")
        self.train_dataset = AbnormalDatasetGradientsTrain(args)
        self.test_dataset = AbnormalDatasetGradientsTest(args)
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Testing samples: {len(self.test_dataset)}")
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=False
        )
        
        # 创建本地MAE模型
        print("Creating local MAE model...")
        if args.dataset == 'avenue':
            self.model = LocalMaskedAutoencoderViT(
                patch_size=16,
                in_chans=9,
                out_chans=4,
                img_size=args.input_size,
                embed_dim=256,
                depth=3,
                num_heads=4,
                decoder_embed_dim=128,
                decoder_depth=3,
                decoder_num_heads=4,
                mlp_ratio=2,
                norm_layer=torch.nn.LayerNorm,
                norm_pix_loss=args.norm_pix_loss,
                use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                abnormal_score_func=args.abnormal_score_func,
                masking_method=args.masking_method,
                grad_weighted_loss=args.grad_weighted_rec_loss
            ).float()
        else:
            self.model = LocalMaskedAutoencoderViT(
                patch_size=8,
                in_chans=9,
                out_chans=4,
                img_size=args.input_size,
                embed_dim=256,
                depth=3,
                num_heads=4,
                decoder_embed_dim=128,
                decoder_depth=3,
                decoder_num_heads=4,
                mlp_ratio=2,
                norm_layer=torch.nn.LayerNorm,
                norm_pix_loss=args.norm_pix_loss,
                use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                abnormal_score_func=args.abnormal_score_func,
                masking_method=args.masking_method,
                grad_weighted_loss=args.grad_weighted_rec_loss
            ).float()
        
        self.model.to(self.device)
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # 创建优化器
        self._create_optimizer()
        
        # 创建学习率调度器
        self._create_scheduler()
        
        # 最佳模型跟踪
        self.best_micro_auc = 0.0
        self.best_macro_auc = 0.0
        self.best_epoch = 0
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
    def _create_optimizer(self):
        """创建优化器"""
        # 使用与联邦学习相同的优化器设置
        param_groups = self._param_groups_weight_decay(self.model, self.args.weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups, 
            lr=self.args.lr, 
            betas=(0.9, 0.95)
        )
        print(f"Optimizer created: {type(self.optimizer).__name__}")
        
    def _create_scheduler(self):
        """创建学习率调度器"""
        # 余弦退火调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.args.epochs,
            eta_min=self.args.lr * 0.01
        )
        
    def _param_groups_weight_decay(self, model, weight_decay):
        """参数分组（参考timm的optim_factory）"""
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # 对于LayerNorm和bias参数，不应用权重衰减
            if name.endswith(".bias") or name.endswith(".norm.weight") or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return param_groups
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 计算每5000个样本对应的batch数
        samples_per_print = 5000
        batches_per_print = samples_per_print // self.args.batch_size
        batches_per_print = max(1, batches_per_print)  # 至少每1个batch打印一次
        
        for batch_idx, (samples, grad_mask, targets) in enumerate(self.train_loader):
            samples = samples.to(self.device)
            grad_mask = grad_mask.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播 - 使用位置参数调用
            loss, _, _ = self.model(samples, targets, grad_mask, self.args.mask_ratio)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印训练进度 - 修改为每5000个样本打印一次
            if batch_idx % batches_per_print == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                current_samples = batch_idx * len(samples)
                total_samples = len(self.train_loader.dataset)
                progress_percent = 100. * batch_idx / len(self.train_loader)
                print(f'Epoch: {epoch} [{current_samples}/{total_samples} '
                    f'({progress_percent:.0f}%)]\t'
                    f'Loss: {loss.item():.6f}\tLR: {current_lr:.6f}')
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def evaluate(self):
        """在测试集上评估模型"""
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
                
                # MAE前向传播 - 测试模式返回4个值
                loss, _, _, recon_error = self.model(samples, targets, grads, self.args.mask_ratio)
                
                # 处理recon_error
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
                pred = filt(pred, range=900, mu=282)
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

        print(f"Evaluation - MicroAUC: {micro_auc:.4f}, MacroAUC: {macro_auc:.4f}")
        return micro_auc, macro_auc
    
    def save_checkpoint(self, epoch, micro_auc, macro_auc, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'micro_auc': micro_auc,
            'macro_auc': macro_auc,
            'args': self.args,
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.args.output_dir, "checkpoint-best-local.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Best model saved at epoch {epoch} with MicroAUC: {micro_auc:.4f}")
        else:
            checkpoint_path = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
        
        # 也保存一个最新的检查点
        latest_path = os.path.join(self.args.output_dir, "checkpoint-latest.pth")
        torch.save(checkpoint, latest_path)
    
    def train(self):
        """主训练循环"""
        print("Starting local training...")
        start_time = time.time()
        
        # 创建TensorBoard日志记录器
        log_writer = SummaryWriter(log_dir=self.args.output_dir)
        
        for epoch in range(self.args.start_epoch, self.args.epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.args.epochs} ===")
            
            # 训练一个epoch
            avg_loss = self.train_epoch(epoch)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 评估模型
            if (epoch + 1) % 1 == 0:
                micro_auc, macro_auc = self.evaluate()
                
                # 记录到TensorBoard
                log_writer.add_scalar('train/loss', avg_loss, epoch)
                log_writer.add_scalar('eval/micro_auc', micro_auc, epoch)
                log_writer.add_scalar('eval/macro_auc', macro_auc, epoch)
                
                # 保存最佳模型
                if micro_auc > self.best_micro_auc:
                    self.best_micro_auc = micro_auc
                    self.best_macro_auc = macro_auc
                    self.best_epoch = epoch + 1
                    self.save_checkpoint(epoch + 1, micro_auc, macro_auc, is_best=True)
                
                # 定期保存检查点
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(epoch + 1, micro_auc, macro_auc)
                
                # 记录到日志文件
                log_stats = {
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'micro_auc': micro_auc,
                    'macro_auc': macro_auc,
                    'best_micro_auc': self.best_micro_auc,
                    'best_macro_auc': self.best_macro_auc,
                    'best_epoch': self.best_epoch,
                }
                
                log_file = os.path.join(self.args.output_dir, "log_local.txt")
                with open(log_file, mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
            
            print(f"Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}")
        
        # 训练完成
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Total training time: {total_time_str}")
        print(f"Best MicroAUC: {self.best_micro_auc:.4f} at epoch {self.best_epoch}")
        print(f"Best MacroAUC: {self.best_macro_auc:.4f}")
        print(f"{'='*60}")
        
        # 关闭TensorBoard写入器
        log_writer.close()
        
        # 保存最终模型
        final_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_micro_auc': self.best_micro_auc,
            'best_macro_auc': self.best_macro_auc,
            'best_epoch': self.best_epoch,
            'args': self.args,
        }
        
        final_path = os.path.join(self.args.output_dir, "final-local-model.pth")
        torch.save(final_checkpoint, final_path)
        print(f"Final model saved to {final_path}")


def main_local(args):
    """本地训练主函数"""
    print('Local Training Job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    # 创建训练器
    trainer = LocalMAETrainer(args)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local MAE Training for Video Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='shanghai', choices=['avenue', 'shanghai'], 
                        help='Dataset name (avenue or shanghai)')
    parser.add_argument('--local', action='store_true', default=True, help='Use local training')
    
    # 添加本地训练特定参数
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Masking ratio')
    
    args = parser.parse_args()
    
    # 获取配置
    if args.dataset == 'avenue':
        config = get_configs_avenue()
    else:
        config = get_configs_shanghai()
    
    # 覆盖本地训练特定的参数
    config.epochs = args.epochs
    config.lr = args.lr
    config.weight_decay = args.weight_decay
    config.batch_size = args.batch_size
    config.mask_ratio = args.mask_ratio
    config.output_dir = args.output_dir
    
    # 确保输出目录存在
    if config.output_dir:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 运行本地训练
    main_local(config)