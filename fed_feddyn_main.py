# fed_baseline_main.py
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import ml_collections
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from configs.fed_baseline_configs import get_fed_baseline_configs_avenue, get_fed_baseline_configs_shanghai
from data.test_dataset import AbnormalDatasetGradientsTest
from data.train_dataset import AbnormalDatasetGradientsTrain
from model.mae_vit import MaskedAutoencoderViT
from fed.fed_feddyn_trainer import FedMAEBaselineTrainer
from fed.fed_feddyn_server import FedMAEBaselineServer
from data.data_split import split_datasets_with_min_proportion


def save_checkpoint(server, clients, round_idx, args, filepath):
    """保存检查点"""
    checkpoint = {
        'global_params': server.broadcast_global_params(),
        'local_states': {client_id: client.get_local_state() for client_id, client in clients.items()},
        'round': round_idx,
        'args': args,
        'timestamp': time.time()
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath} at round {round_idx}")


def save_best_model_checkpoint(global_params, local_states, micro_auc, macro_auc, round_idx, args, filepath):
    """保存最佳模型检查点"""
    checkpoint = {
        'global_params': global_params,
        'local_states': local_states,
        'micro_auc': micro_auc,
        'macro_auc': macro_auc,
        'round': round_idx,
        'args': args,
        'timestamp': time.time()
    }
    torch.save(checkpoint, filepath)
    print(f"Best model checkpoint saved to {filepath}")


def main_fed_baseline(args):
    print('Fed Baseline Job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    device = args.device
    log_writer = SummaryWriter(log_dir=args.output_dir)

    # 加载训练和测试数据集
    dataset_train = AbnormalDatasetGradientsTrain(args)
    dataset_test = AbnormalDatasetGradientsTest(args)
    print(f"Total training samples: {len(dataset_train)}")
    print(f"Test samples: {len(dataset_test)}")
    
    # 设置最小数据比例阈值
    min_proportion = getattr(args, 'min_data_proportion', 0.15)
    print(f"使用最小数据比例阈值: {min_proportion:.1%}")
    
    # 使用带最小比例阈值的数据划分
    client_train_datasets, client_test_datasets, client_proportions = split_datasets_with_min_proportion(
        args, dataset_train, dataset_test, args.fed_num_clients, 
        min_proportion=min_proportion,
        seed=getattr(args, 'seed', 5555)
    )
    
    # 保存划分信息
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        split_info = {
            'client_proportions': client_proportions,
            'client_train_sizes': [len(ds) for ds in client_train_datasets],
            'client_test_sizes': [len(ds) for ds in client_test_datasets],
            'total_train_samples': len(dataset_train),
            'total_test_samples': len(dataset_test),
            'seed': getattr(args, 'seed', 5555)
        }
        
        split_info_path = os.path.join(args.output_dir, "data_split_info.json")
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"\n划分信息已保存到: {split_info_path}")

    # 在main_fed_baseline函数中，修改模型创建部分：

    # 定义基础MAE模型 - 使用联邦学习基础版本
    if args.dataset == 'avenue':
        from model.fed_baseline_mae_model import FedBaselineMaskedAutoencoderViT
        base_mae = FedBaselineMaskedAutoencoderViT(
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
        from model.fed_baseline_mae_model import FedBaselineMaskedAutoencoderViT
        base_mae = FedBaselineMaskedAutoencoderViT(
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

    base_mae.to(device)
    
    # 创建联邦学习服务器
    server = FedMAEBaselineServer(base_mae, args)
    
    # 创建客户端训练器
    clients = {}
    print("\n" + "="*60)
    print("创建基础联邦学习客户端")
    print("="*60)
    
    for i in range(args.fed_num_clients):
        client_id = f'client_{i}'
        client_train_dataset = client_train_datasets[i]
        client_test_dataset = client_test_datasets[i]

        # 检查数据集是否为空
        if len(client_train_dataset) == 0:
            raise ValueError(f"客户端 {client_id} 的训练数据集为空！")
        if len(client_test_dataset) == 0:
            raise ValueError(f"客户端 {client_id} 的测试数据集为空！")
        
        print(f"\n创建客户端 {client_id}:")
        print(f"  分配比例: {client_proportions[i]:.2%}")
        print(f"  训练样本数: {len(client_train_dataset)} ({len(client_train_dataset)/len(dataset_train):.2%})")
        print(f"  测试样本数: {len(client_test_dataset)} ({len(client_test_dataset)/len(dataset_test):.2%})")
        
        # 创建数据加载器
        client_data_loader = torch.utils.data.DataLoader(
            client_train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=True
        )
        
        client_test_loader = torch.utils.data.DataLoader(
            client_test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False
        )
        
        clients[client_id] = FedMAEBaselineTrainer(client_id, base_mae, args, client_data_loader, client_test_loader)
        print(f"  可训练参数: {len(clients[client_id].model.get_trainable_params())}")
    
    print(f"\n所有客户端创建完成,开始基础联邦学习训练...")

    # 联邦学习训练循环
    do_federated_training(args, clients, server, device, log_writer)


def do_federated_training(args, clients, server, device, log_writer):
    print("Starting Federated Baseline Training...")
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    best_avg_micro_auc = 0.0
    best_avg_macro_auc = 0.0
    best_round = 0

    best_global_params = None
    best_local_states = {}
    
    for round_idx in range(args.fed_global_rounds):
        print(f"\n=== Federated Baseline Round {round_idx + 1}/{args.fed_global_rounds} ===")
        round_start_time = time.time()
        
        # 1. 服务器广播全局参数
        global_params = server.broadcast_global_params()
        print(f"Broadcasted {len(global_params)} global parameters")
        
        # 2. 客户端本地训练
        client_updates = {}
        client_weights = {}
        client_losses = {}
        
        for client_id, client in clients.items():
            print(f"Training {client_id}...")
            client_start_time = time.time()
            
            # 加载全局参数
            client.load_global_params(global_params)
            
            # 本地训练
            avg_loss = client.local_train(args.fed_local_epochs)
            client_losses[client_id] = avg_loss
            
            # 获取客户端更新
            client_updates[client_id] = client.get_global_updates()
            
            # 设置客户端权重（根据数据量）
            client_weights[client_id] = len(client.data_loader.dataset)
            
            client_time = time.time() - client_start_time
            print(f"Client {client_id} completed in {client_time:.2f}s, Loss: {avg_loss:.4f}")
        
        # 3. 服务器聚合更新
        server.aggregate_updates(client_updates, client_weights)
        
        # 4. 评估全局模型（每轮都评估）
        print("Evaluating each client's model on their local test set...")
        client_micro_aucs = []
        client_macro_aucs = []
        
        for client_id, client in clients.items():
            micro_auc, macro_auc = client.evaluate()
            client_micro_aucs.append(micro_auc)
            client_macro_aucs.append(macro_auc)
        
        # 计算平均AUC
        avg_micro_auc = np.mean(client_micro_aucs)
        avg_macro_auc = np.mean(client_macro_aucs)
        
        # 5. 更新最佳模型
        if avg_micro_auc > best_avg_micro_auc:
            best_avg_micro_auc = avg_micro_auc
            best_avg_macro_auc = avg_macro_auc
            best_round = round_idx + 1
            best_global_params = server.broadcast_global_params()

            # 保存所有客户端的最佳本地状态
            best_local_states = {}
            for client_id, client in clients.items():
                best_local_states[client_id] = client.get_local_state()
            
            # 保存最佳模型检查点
            best_model_path = os.path.join(args.output_dir, "checkpoint-best-fed-baseline.pth")
            save_best_model_checkpoint(
                best_global_params, 
                best_local_states, 
                best_avg_micro_auc, 
                best_avg_macro_auc, 
                best_round, 
                args, 
                best_model_path
            )
            print(f"New best model saved at round {best_round} with average Micro AUC: {best_avg_micro_auc:.4f}")
        
        # 记录日志
        round_time = time.time() - round_start_time
        log_stats = {
            'round': round_idx + 1,
            'avg_client_loss': sum(client_losses.values()) / len(client_losses),
            'client_micro_aucs': client_micro_aucs,
            'client_macro_aucs': client_macro_aucs,
            'avg_micro_auc': avg_micro_auc,
            'avg_macro_auc': avg_macro_auc,
            'best_avg_micro_auc': best_avg_micro_auc,
            'best_avg_macro_auc': best_avg_macro_auc,
            'best_round': best_round,
            'round_time': round_time
        }
        
        # 写入TensorBoard
        if log_writer is not None:
            log_writer.add_scalar('fed_baseline/avg_client_loss', log_stats['avg_client_loss'], round_idx + 1)
            log_writer.add_scalar('fed_baseline/avg_micro_auc', avg_micro_auc, round_idx + 1)
            log_writer.add_scalar('fed_baseline/avg_macro_auc', avg_macro_auc, round_idx + 1)
            log_writer.add_scalar('fed_baseline/best_avg_micro_auc', best_avg_micro_auc, round_idx + 1)
            log_writer.add_scalar('fed_baseline/round_time', round_time, round_idx + 1)
            
            # 记录每个客户端的AUC
            for i, (micro_auc, macro_auc) in enumerate(zip(client_micro_aucs, client_macro_aucs)):
                log_writer.add_scalar(f'fed_baseline/client_{i}_micro_auc', micro_auc, round_idx + 1)
                log_writer.add_scalar(f'fed_baseline/client_{i}_macro_auc', macro_auc, round_idx + 1)
        
        # 定期保存检查点(每10轮)
        if (round_idx + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-round-{round_idx+1}.pth")
            save_checkpoint(server, clients, round_idx + 1, args, checkpoint_path)
        
        # 写入日志文件
        if args.output_dir:
            with open(os.path.join(args.output_dir, "log_fed_baseline.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        print(f"Round {round_idx + 1} completed in {round_time:.2f}s")
        print(f"Client Micro AUCs: {[f'{auc:.4f}' for auc in client_micro_aucs]}")
        print(f"Client Macro AUCs: {[f'{auc:.4f}' for auc in client_macro_aucs]}")
        print(f"Average Micro AUC: {avg_micro_auc:.4f}, Average Macro AUC: {avg_macro_auc:.4f}")
        print(f"Best Average Micro AUC: {best_avg_micro_auc:.4f} (Round {best_round})")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\nFederated Baseline Training completed in {total_time_str}')
    print(f'Best average micro AUC: {best_avg_micro_auc:.4f} at round {best_round}') 

    # 最终保存最佳模型
    final_best_model_path = os.path.join(args.output_dir, "final-best-fed-baseline-model.pth")
    save_best_model_checkpoint(
        best_global_params, 
        best_local_states, 
        best_avg_micro_auc, 
        best_avg_macro_auc, 
        best_round, 
        args, 
        final_best_model_path
    ) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='shanghai')
    parser.add_argument('--fed_baseline', action='store_true', default=True, help='Use federated baseline learning')
    parser.add_argument('--mu', type=float, default=0.1, help='Proximal term coefficient for FedProx (0.0 for FedAvg)')
    args = parser.parse_args()
    
    if args.dataset == 'avenue':
        config = get_fed_baseline_configs_avenue()
    else:
        config = get_fed_baseline_configs_shanghai()
    
    if config.output_dir:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    main_fed_baseline(config)