# main_fed_mae.py
import argparse
import datetime
import json
import os
import time
from pathlib import Path
import ml_collections

from timm.optim import optim_factory
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter

from data.test_dataset import AbnormalDatasetGradientsTest
from data.train_dataset import AbnormalDatasetGradientsTrain
from model.mae_vit import MaskedAutoencoderViT
from fed.fed_mae_trainer import FedMAETrainer
from fed.fed_server import FedMAEServer
from data.data_save import save_best_model_checkpoint
from data.data_save import save_checkpoint
import torch
from util import misc
import numpy as np
from util.abnormal_utils import filt
import sklearn.metrics as metrics
from configs.fed_configs import get_configs_avenue, get_configs_shanghai
import json
from data.data_split import split_datasets_with_min_proportion
import random

# 场景到前缀的映射
SCENE_PREFIX_MAP = {
    'street': ['02', '06', '08', '09', '13'],   # 训练集多一个13，测试集无13但代码会自动忽略不存在的
    'park': ['01', '04', '10', '12'],
    'zebra_crossing': ['03', '07'],
    'road': ['05', '11']
}

from torch.utils.data import Subset, Dataset
from collections import defaultdict

def split_datasets_by_scene(dataset_train: Dataset, dataset_test: Dataset):
    """
    根据场景划分训练集和测试集，返回4个客户端的数据子集。
    
    Returns:
        client_train_datasets: list of Subset, 每个元素对应一个客户端的训练子集
        client_test_datasets: list of Subset, 每个元素对应一个客户端的测试子集
        client_names: list of str, 客户端名称（场景名）
        client_proportions: list of float, 每个客户端训练样本占比（基于训练集总数）
    """
    # 建立前缀 -> 场景的映射（用于快速查找）
    prefix_to_scene = {}
    for scene, prefixes in SCENE_PREFIX_MAP.items():
        for p in prefixes:
            prefix_to_scene[p] = scene

    # 辅助函数：从路径中提取视频文件夹前缀（前两位数字）
    def get_prefix_from_path(path: str) -> str:
        # 路径格式类似: .../01_0014/0001.png
        folder_name = path.split('/')[-2]   # 例如 '01_0014'
        prefix = folder_name.split('_')[0]  # 取 '01'
        return prefix

    # 辅助函数：根据数据集和索引获取该样本所属的场景
    def get_scene_for_index(dataset, idx, is_train=True):
        if is_train:
            # 训练集使用 self.data 获取路径
            path = dataset.data[idx]
        else:
            path = dataset.data[idx]
        prefix = get_prefix_from_path(path)
        scene = prefix_to_scene.get(prefix)
        if scene is None:
            raise ValueError(f"未知的前缀 '{prefix}'，来自路径: {path}")
        return scene

    # 收集每个场景的训练索引
    scene_to_train_indices = defaultdict(list)
    for idx in range(len(dataset_train)):
        scene = get_scene_for_index(dataset_train, idx, is_train=True)
        scene_to_train_indices[scene].append(idx)

    # 收集每个场景的测试索引
    scene_to_test_indices = defaultdict(list)
    for idx in range(len(dataset_test)):
        scene = get_scene_for_index(dataset_test, idx, is_train=False)
        scene_to_test_indices[scene].append(idx)

    # 按照固定顺序生成客户端列表
    client_names = list(SCENE_PREFIX_MAP.keys())  # ['street', 'park', 'zebra_crossing', 'road']
    client_train_datasets = []
    client_test_datasets = []
    client_proportions = []

    total_train = len(dataset_train)
    for scene in client_names:
        train_indices = scene_to_train_indices.get(scene, [])
        test_indices = scene_to_test_indices.get(scene, [])
        if len(train_indices) == 0:
            print(f"警告: 场景 {scene} 在训练集中没有样本！")
        if len(test_indices) == 0:
            print(f"警告: 场景 {scene} 在测试集中没有样本！")
        
        client_train_datasets.append(Subset(dataset_train, train_indices))
        client_test_datasets.append(Subset(dataset_test, test_indices))
        client_proportions.append(len(train_indices) / total_train if total_train > 0 else 0)

    return client_train_datasets, client_test_datasets, client_names, client_proportions


def main_fed(args):
    print('Federated Learning Job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    device = args.device
    log_writer = SummaryWriter(log_dir=args.output_dir)

    # 加载训练和测试数据集
    dataset_train = AbnormalDatasetGradientsTrain(args)
    dataset_test = AbnormalDatasetGradientsTest(args)
    print(f"Total training samples: {len(dataset_train)}")
    print(f"Test samples: {len(dataset_test)}")
    
    ''' 随机划分
    # 设置最小数据比例阈值（例如10%）
    min_proportion = getattr(args, 'min_data_proportion', 0.15)
    print(f"使用最小数据比例阈值: {min_proportion:.1%}")
    
    # 使用带最小比例阈值的数据划分
    client_train_datasets, client_test_datasets, client_proportions = split_datasets_with_min_proportion(
        args, dataset_train, dataset_test, args.fed_num_clients, 
        min_proportion=min_proportion,
        seed=getattr(args, 'seed', 42)
    )'''

    # 按场景划分数据（固定4个客户端）
    client_train_datasets, client_test_datasets, client_names, client_proportions = split_datasets_by_scene(
        dataset_train, dataset_test
    )

    # 将客户端数量设置为实际划分出的场景数（应为4）
    args.fed_num_clients = len(client_train_datasets)
    print(f"按场景划分为 {args.fed_num_clients} 个客户端: {client_names}")
    print("各客户端训练样本比例:")
    for name, prop, size in zip(client_names, client_proportions, [len(ds) for ds in client_train_datasets]):
        print(f"  {name}: {prop:.2%} ({size} 样本)")

    # 可选：保存划分信息时记录场景名
    split_info = {
        'client_names': client_names,
        'client_proportions': client_proportions,
        'client_train_sizes': [len(ds) for ds in client_train_datasets],
        'client_test_sizes': [len(ds) for ds in client_test_datasets],
        'total_train_samples': len(dataset_train),
        'total_test_samples': len(dataset_test),
        'split_method': 'scene_based'
    }

    
    # 保存划分信息
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        split_info = {
            'client_proportions': client_proportions,
            'client_train_sizes': [len(ds) for ds in client_train_datasets],
            'client_test_sizes': [len(ds) for ds in client_test_datasets],
            'total_train_samples': len(dataset_train),
            'total_test_samples': len(dataset_test),
            'seed': getattr(args, 'seed', 42)
        }
        
        split_info_path = os.path.join(args.output_dir, "data_split_info.json")
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"\n划分信息已保存到: {split_info_path}")

    # 定义基础MAE模型
    if args.dataset == 'avenue':
        base_mae = MaskedAutoencoderViT(
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
        base_mae = MaskedAutoencoderViT(
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
    server = FedMAEServer(base_mae, args)
    
    # 创建客户端训练器
    clients = {}
    print("\n" + "="*60)
    print("创建联邦学习客户端")
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
        
        clients[client_id] = FedMAETrainer(client_id, base_mae, args, client_data_loader, client_test_loader)
        print(f"  可训练参数: {len(clients[client_id].model.get_trainable_params())}")
    
    print(f"\n所有客户端创建完成,开始联邦学习训练...")

    # 联邦学习训练循环
    do_federated_training(args, clients, server, device, log_writer)

'''联邦学习训练循环'''
def do_federated_training(args, clients, server, device, log_writer):
    print("Starting Federated Training...")
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有客户端ID
    all_client_ids = list(clients.keys())
    total_clients = len(all_client_ids)
    clients_per_round = 3  # 每轮通信随机抽取3个客户端
    
    print(f"Total clients: {total_clients}, Clients per round: {clients_per_round}")
    
    best_avg_micro_auc = 0.0
    best_avg_macro_auc = 0.0
    best_round = 0
    best_global_params = None
    best_local_states = {}
    
    for round_idx in range(args.fed_global_rounds):
        print(f"\n=== Federated Learning Round {round_idx + 1}/{args.fed_global_rounds} ===")
        round_start_time = time.time()
        
        # 1. 随机抽取客户端（用于本轮训练）
        selected_clients = random.sample(all_client_ids, clients_per_round)
        print(f"Selected clients for round {round_idx + 1}: {selected_clients}")
        
        # 2. 服务器广播第t轮全局模型
        if round_idx == 0:
            # 第一轮使用初始全局参数
            global_params = server.broadcast_global_params()
        else:
            # 后续轮次使用上一轮聚合后的全局参数
            global_params = server.global_params  # 直接获取服务器保存的参数
        
        print(f"Broadcasted round {round_idx} global parameters")
        
        # 3. 所有客户端加载第t轮全局模型
        for client_id in all_client_ids:
            clients[client_id].load_global_params(global_params)
        
        # 4. 被采样的客户端进行本地训练
        client_updates = {}
        client_weights = {}
        client_losses = {}
        
        for client_id in selected_clients:
            print(f"Training selected client {client_id}...")
            client_start_time = time.time()
            
            client = clients[client_id]
            
            # 本地训练，得到第t轮个性化模型
            avg_loss = client.local_train(args.fed_local_epochs)
            client_losses[client_id] = avg_loss
            
            # 获取客户端更新（个性化模型与全局模型的差异）
            client_updates[client_id] = client.get_global_updates()
            
            # 设置客户端权重（根据数据量）
            client_weights[client_id] = len(client.data_loader.dataset)
            
            client_time = time.time() - client_start_time
            print(f"Client {client_id} completed in {client_time:.2f}s, Loss: {avg_loss:.4f}")
        
        # 5. 所有客户端进行评估（使用第t轮模型）
        print(f"Evaluating all {total_clients} clients on their local test sets...")
        client_micro_aucs = []
        client_macro_aucs = []
        client_models = []  # 记录使用的模型类型
        
        for client_id in all_client_ids:
            client = clients[client_id]
            was_selected = client_id in selected_clients
            
            # 评估客户端
            # 注意：被选中的客户端使用个性化模型评估
            # 未被选中的客户端使用第t轮全局模型评估
            micro_auc, macro_auc = client.evaluate()
            client_micro_aucs.append(micro_auc)
            client_macro_aucs.append(macro_auc)
            client_models.append("Personalized" if was_selected else "Global")
            
            print(f"Client {client_id} ({'Personalized' if was_selected else 'Global'} model): "
                  f"MicroAUC: {micro_auc:.4f}, MacroAUC: {macro_auc:.4f}")
        
        # 6. 服务器聚合被采样客户端的更新
        if client_updates:  # 确保有更新可聚合
            server.aggregate_updates(client_updates, client_weights)
            print(f"Aggregated updates from {len(selected_clients)} selected clients")
        else:
            print("No client updates to aggregate")
        
        # 7. 计算平均AUC（基于所有客户端的评估结果）
        avg_micro_auc = np.mean(client_micro_aucs)
        avg_macro_auc = np.mean(client_macro_aucs)
        
        # 8. 更新最佳模型（基于全局模型的评估结果）
        # 注意：我们以全局模型的性能为准，所以计算未被选中客户端的平均AUC作为全局模型性能
        global_client_aucs = [auc for auc, model_type in zip(client_micro_aucs, client_models) 
                             if model_type == "Global"]
        global_avg_micro_auc = np.mean(global_client_aucs) if global_client_aucs else avg_micro_auc
        
        if global_avg_micro_auc > best_avg_micro_auc:
            best_avg_micro_auc = global_avg_micro_auc
            best_avg_macro_auc = np.mean([auc for auc, model_type in zip(client_macro_aucs, client_models) 
                                         if model_type == "Global"])
            best_round = round_idx + 1
            best_global_params = server.broadcast_global_params()  # 保存当前轮次的全局参数
            
            # 保存所有客户端的最佳本地状态
            best_local_states = {}
            for client_id in all_client_ids:
                best_local_states[client_id] = clients[client_id].get_local_state()
            
            # 保存最佳模型检查点
            best_model_path = os.path.join(args.output_dir, "checkpoint-best-fed.pth")
            save_best_model_checkpoint(
                best_global_params, 
                best_local_states, 
                best_avg_micro_auc, 
                best_avg_macro_auc, 
                best_round, 
                args, 
                best_model_path
            )
            print(f"New best global model saved at round {best_round} with average Micro AUC: {best_avg_micro_auc:.4f}")
        
        # 9. 记录日志
        round_time = time.time() - round_start_time
        log_stats = {
            'round': round_idx + 1,
            'selected_clients': selected_clients,
            'avg_selected_client_loss': sum(client_losses.values()) / len(client_losses) if client_losses else 0,
            'client_micro_aucs': client_micro_aucs,
            'client_macro_aucs': client_macro_aucs,
            'client_models': client_models,  # 记录使用的模型类型
            'global_avg_micro_auc': global_avg_micro_auc,
            'global_avg_macro_auc': np.mean([auc for auc, model_type in zip(client_macro_aucs, client_models) 
                                            if model_type == "Global"]),
            'avg_micro_auc': avg_micro_auc,
            'avg_macro_auc': avg_macro_auc,
            'best_avg_micro_auc': best_avg_micro_auc,
            'best_avg_macro_auc': best_avg_macro_auc,
            'best_round': best_round,
            'round_time': round_time
        }
        
        # 写入TensorBoard
        if log_writer is not None:
            log_writer.add_scalar('fed/avg_selected_client_loss', log_stats['avg_selected_client_loss'], round_idx + 1)
            log_writer.add_scalar('fed/avg_micro_auc', avg_micro_auc, round_idx + 1)
            log_writer.add_scalar('fed/avg_macro_auc', avg_macro_auc, round_idx + 1)
            log_writer.add_scalar('fed/global_avg_micro_auc', global_avg_micro_auc, round_idx + 1)
            log_writer.add_scalar('fed/best_avg_micro_auc', best_avg_micro_auc, round_idx + 1)
            log_writer.add_scalar('fed/selected_clients_ratio', len(selected_clients)/total_clients, round_idx + 1)
            log_writer.add_scalar('fed/round_time', round_time, round_idx + 1)
            
            # 记录每个客户端的AUC
            for i, (micro_auc, macro_auc, model_type) in enumerate(zip(client_micro_aucs, client_macro_aucs, client_models)):
                client_id = all_client_ids[i]
                log_writer.add_scalar(f'fed/client_{i}_micro_auc', micro_auc, round_idx + 1)
                log_writer.add_scalar(f'fed/client_{i}_macro_auc', macro_auc, round_idx + 1)
                log_writer.add_scalar(f'fed/client_{i}_model_type', 1 if model_type == "Personalized" else 0, round_idx + 1)
        
        # 10. 定期保存检查点(每10轮)
        if (round_idx + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-round-{round_idx+1}.pth")
            save_checkpoint(server, clients, round_idx + 1, args, checkpoint_path)
        
        # 11. 写入日志文件
        if args.output_dir:
            with open(os.path.join(args.output_dir, "log_fed.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        print(f"Round {round_idx + 1} completed in {round_time:.2f}s")
        print(f"Selected clients: {selected_clients}")
        print(f"Global model average Micro AUC: {global_avg_micro_auc:.4f}")
        print(f"All clients average Micro AUC: {avg_micro_auc:.4f}")
        print(f"Best Global Model Micro AUC: {best_avg_micro_auc:.4f} (Round {best_round})")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Federated Training completed in {total_time_str}')
    print(f'Best global model micro AUC: {best_avg_micro_auc:.4f} at round {best_round}') 

    # 最终保存最佳模型
    final_best_model_path = os.path.join(args.output_dir, "final-best-fed-model.pth")
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
    parser.add_argument('--fed', action='store_true', default=True, help='Use federated learning')
    parser.local_parts = ['adapter']
    args = parser.parse_args()
    
    if args.dataset == 'avenue':
        config = get_configs_avenue()
    else:
        config = get_configs_shanghai()
    
    if config.output_dir:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.fed:
        main_fed(config)
    else:
        print("Single machine training not implemented in this script")