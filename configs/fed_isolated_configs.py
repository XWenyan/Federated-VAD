# configs/fed_isolated_configs.py
import ml_collections

def get_fed_isolated_configs_avenue():
    """孤立客户端联邦学习的Avenue数据集配置"""
    config = ml_collections.ConfigDict()
    
    # 训练参数
    config.batch_size = 100
    config.epochs = 50
    config.mask_ratio = 0.5
    config.masking_method = "random_masking"
    config.output_dir = "experiments/fed_isolated_avenue"
    
    # 模型参数
    config.abnormal_score_func = ['L2', 'L2']
    config.grad_weighted_rec_loss = True
    config.model = "mae_vit"
    config.input_size = (320, 640)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    
    # 训练设置
    config.run_type = 'train'
    config.resume = False
    config.seed = 42
    config.start_epoch = 0
    config.print_freq = 50
    
    # 优化器参数
    config.weight_decay = 0.05
    config.lr = 1e-4
    
    # 数据集参数
    config.dataset = "avenue"
    config.avenue_path = "/mnt/private/xwy/aed-mae/data/avenue"
    config.avenue_gt_path = "/mnt/private/xwy/aed-mae/data/avenue/Avenue_gt"
    config.percent_abnormal = 0.3
    config.input_3d = True
    config.device = "cuda"
    
    # 数据加载参数
    config.num_workers = 5
    config.pin_mem = False
    
    # 联邦学习参数
    config.fed_global_rounds = 50
    config.fed_local_epochs = 10
    config.fed_num_clients = 3
    config.min_data_proportion = 0.15
    
    return config

def get_fed_isolated_configs_shanghai():
    """孤立客户端联邦学习的Shanghai数据集配置"""
    config = ml_collections.ConfigDict()
    
    # 训练参数
    config.batch_size = 100
    config.epochs = 10
    config.mask_ratio = 0.5
    config.masking_method = "random_masking"
    config.output_dir = "experiments/fed_isolated_shanghai"
    
    # 模型参数
    config.abnormal_score_func = ['L2', 'L2']
    config.grad_weighted_rec_loss = True
    config.model = "mae_vit"
    config.input_size = (160, 320)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    
    # 训练设置
    config.run_type = 'train'
    config.resume = False
    config.seed = 42
    config.start_epoch = 0
    config.print_freq = 50
    
    # 优化器参数
    config.weight_decay = 0.05
    config.lr = 1e-4
    
    # 数据集参数
    config.dataset = "shanghai"
    config.shanghai_path = "/mnt/private/xwy/aed-mae/data/shanghaitech_AED"
    config.shanghai_gt_path = "/mnt/private/xwy/aed-mae/data/shanghaitech_AED/Shanghai_gt"
    config.percent_abnormal = 0.5
    config.input_3d = True
    config.device = "cuda"
    
    # 数据加载参数
    config.num_workers = 5
    config.pin_mem = False
    
    # 联邦学习参数
    config.fed_global_rounds = 10
    config.fed_local_epochs = 5
    config.fed_num_clients = 10
    config.min_data_proportion = 0.08
    
    return config