import torch
from fed.fed_server import FedMAEServer
import time
import os
from model.mae_vit import MaskedAutoencoderViT

"""保存最佳模型检查点"""
def save_best_model_checkpoint(global_params, local_states, micro_auc, macro_auc, round_idx, args, filepath):
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

"""保存常规检查点"""
def save_checkpoint(server, clients, round_idx, args, filepath):
    checkpoint = {
        'global_params': server.broadcast_global_params(),
        'local_states': {client_id: client.get_local_state() for client_id, client in clients.items()},
        'round': round_idx,
        'args': args,
        'timestamp': time.time()
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath} at round {round_idx}")

"""加载检查点"""
def load_checkpoint(filepath, server, clients, device):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        
        # 加载全局参数到服务器
        server.global_params = checkpoint['global_params']
        
        # 加载本地状态到各个客户端
        for client_id, local_state in checkpoint['local_states'].items():
            if client_id in clients:
                clients[client_id].load_local_state(local_state)
        
        print(f"Checkpoint loaded from {filepath}, round {checkpoint['round']}")
        return checkpoint['round'], checkpoint.get('micro_auc', 0.0), checkpoint.get('macro_auc', 0.0)
    else:
        print(f"Checkpoint {filepath} not found")
        return 0, 0.0, 0.0


# 注：暂未使用
"""加载训练好的联邦学习模型"""
def load_trained_fed_model(args, model_path=None):
    if model_path is None:
        model_path = os.path.join(args.output_dir, "final-best-fed-model.pth")
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found")
        return None, None, 0.0, 0.0
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=args.device)
    
    # 创建基础MAE模型
    if args.dataset == 'avenue':
        base_mae = MaskedAutoencoderViT(
            patch_size=16,
            in_chans=9,
            out_chans=4,
            img_size=args.input_size,
            embed_dim=256,
            depth=12,
            num_heads=8,
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
            depth=12,
            num_heads=8,
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
    
    base_mae.to(args.device)
    
    # 创建联邦学习服务器
    server = FedMAEServer(base_mae, args)
    
    # 加载全局参数
    server.global_params = checkpoint['global_params']
    
    # 创建客户端（用于推理）
    clients = {}
    for client_id, local_state in checkpoint['local_states'].items():
        # 这里简化处理，实际使用时可能需要根据保存的客户端信息重新创建
        from fed.fed_mae_trainer import FedMAETrainer
        # 注意：这里需要传递适当的数据加载器，对于推理可以为None
        client = FedMAETrainer(client_id, base_mae, args, None, None)
        client.load_local_state(local_state)
        clients[client_id] = client
    
    micro_auc = checkpoint.get('micro_auc', 0.0)
    macro_auc = checkpoint.get('macro_auc', 0.0)
    round_idx = checkpoint.get('round', 0)
    
    print(f"Successfully loaded trained federated model from round {round_idx}")
    print(f"Performance: Micro AUC = {micro_auc:.4f}, Macro AUC = {macro_auc:.4f}")
    
    return server, clients, micro_auc, macro_auc