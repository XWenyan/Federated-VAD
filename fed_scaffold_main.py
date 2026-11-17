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
from fed.fed_scaffold_trainer import FedMAEBaselineTrainer
from fed.fed_scaffold_server import FedMAEBaselineServer
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


# fed_baseline_main.py （其余保持不变）

def do_federated_training(args, clients, server, device, log_writer):
    print("Starting Federated SCAFFOLD Training...")
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    best_avg_micro_auc = 0.0
    best_avg_macro_auc = 0.0
    best_round = 0
    best_global_params = None
    best_local_states = {}

    for round_idx in range(args.fed_global_rounds):
        print(f"\n=== Federated SCAFFOLD Round {round_idx + 1}/{args.fed_global_rounds} ===")
        round_start_time = time.time()

        # 1. 服务器广播全局模型参数和全局控制变量
        global_params, global_control = server.broadcast_global_params()
        print(f"Broadcasted {len(global_params)} parameters and control variables")

        # 2. 客户端本地训练并上传更新
        client_updates = {}
        client_weights = {}
        client_losses = {}

        for client_id, client in clients.items():
            print(f"Training {client_id}...")
            client_start_time = time.time()

            # 加载全局模型参数（客户端内部加载）
            client.load_global_params(global_params, global_control)

            # 本地训练（传入服务器控制变量用于梯度修正）
            avg_loss = client.local_train(args.fed_local_epochs, global_control)
            client_losses[client_id] = avg_loss

            # 获取客户端上传的更新 (delta_model, delta_control)
            delta_model, delta_control = client.get_global_updates()
            if delta_model is not None:
                client_updates[client_id] = (delta_model, delta_control)
                client_weights[client_id] = len(client.data_loader.dataset)

            client_time = time.time() - client_start_time
            print(f"Client {client_id} completed in {client_time:.2f}s, Loss: {avg_loss:.4f}")

        # 3. 服务器聚合更新
        server.aggregate_updates(client_updates, client_weights)

        # 4. 评估全局模型（每轮均评估，与之前相同）
        client_micro_aucs = []
        client_macro_aucs = []
        for client_id, client in clients.items():
            micro_auc, macro_auc = client.evaluate()
            client_micro_aucs.append(micro_auc)
            client_macro_aucs.append(macro_auc)

        avg_micro_auc = np.mean(client_micro_aucs)
        avg_macro_auc = np.mean(client_macro_aucs)

        # 5. 更新最佳模型（逻辑不变）
        if avg_micro_auc > best_avg_micro_auc:
            best_avg_micro_auc = avg_micro_auc
            best_avg_macro_auc = avg_macro_auc
            best_round = round_idx + 1
            best_global_params = server.broadcast_global_params()[0]  # 只保存模型参数
            best_local_states = {cid: client.get_local_state() for cid, client in clients.items()}
            best_model_path = os.path.join(args.output_dir, "checkpoint-best-fed-scaffold.pth")
            save_best_model_checkpoint(
                best_global_params, best_local_states,
                best_avg_micro_auc, best_avg_macro_auc, best_round, args, best_model_path
            )
            print(f"New best model at round {best_round} with avg Micro AUC: {best_avg_micro_auc:.4f}")

        # 日志记录（与之前相同）
        round_time = time.time() - round_start_time
        log_stats = {
            'round': round_idx + 1,
            'avg_client_loss': sum(client_losses.values())/len(client_losses),
            'client_micro_aucs': client_micro_aucs,
            'client_macro_aucs': client_macro_aucs,
            'avg_micro_auc': avg_micro_auc,
            'avg_macro_auc': avg_macro_auc,
            'best_avg_micro_auc': best_avg_micro_auc,
            'best_round': best_round,
            'round_time': round_time,
        }
        if log_writer:
            log_writer.add_scalar('fed_scaffold/avg_client_loss', log_stats['avg_client_loss'], round_idx+1)
            log_writer.add_scalar('fed_scaffold/avg_micro_auc', avg_micro_auc, round_idx+1)
            # ... 其他记录类似

        if (round_idx + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-round-{round_idx+1}.pth")
            save_checkpoint(server, clients, round_idx+1, args, checkpoint_path)

        print(f"Round {round_idx+1} completed. Avg Micro AUC: {avg_micro_auc:.4f}, Best: {best_avg_micro_auc:.4f}")

    total_time = time.time() - start_time
    print(f'\nFederated SCAFFOLD Training completed in {str(datetime.timedelta(seconds=int(total_time)))}')
    print(f'Best average micro AUC: {best_avg_micro_auc:.4f} at round {best_round}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='shanghai')
    parser.add_argument('--fed_baseline', action='store_true', default=True, help='Use federated baseline learning')
    
    args = parser.parse_args()
    
    if args.dataset == 'avenue':
        config = get_fed_baseline_configs_avenue()
    else:
        config = get_fed_baseline_configs_shanghai()
    
    if config.output_dir:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    main_fed_baseline(config)