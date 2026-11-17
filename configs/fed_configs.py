import ml_collections

def get_configs_avenue():
    config = ml_collections.ConfigDict()
    config.batch_size = 100
    config.epochs = 200
    config.mask_ratio = 0.5
    config.masking_method = "random_masking"
    config.output_dir = "experiments/avenue_fed_ge_m2" # 编码器本地
    config.abnormal_score_func = ['L2', 'L2']
    config.grad_weighted_rec_loss = True
    config.model = "mae_vit"
    config.input_size = (320, 640)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = 'train'
    config.resume = False
    config.seed = 42
    config.min_data_proportion = 0.08  # 最小数据比例阈值
    config.fed_scene_dim = 64
    
    # 联邦学习参数
    config.fed_global_rounds = 200
    config.fed_local_epochs = 10
    config.fed_num_clients = 10 # 客户端数量
    config.fed_mid_dim = 256
    config.fed_scale = 1.5
    config.fed_vanilla = '-'
    config.fed_prefix_depth = 3
    config.fed_local_parts = ['adapter']
    
    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "avenue"
    config.avenue_path = "/mnt/private/xwy/aed-mae/data/avenue"
    config.avenue_gt_path = "/mnt/private/xwy/aed-mae/data/avenue/Avenue_gt"
    config.percent_abnormal = 0.3
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 5
    config.pin_mem = False

    return config

def get_configs_shanghai():
    config = ml_collections.ConfigDict()
    config.batch_size = 100
    config.mask_ratio = 0.5
    config.masking_method = "random_masking"
    config.output_dir = "experiments/shanghai_fed__ge_m1_sceen" # 系数0.1
    # config.output_dir = "experiments/shanghai_fed_seed42_0.5_new_1_abnormal0.3" # 随机种子 掩码比例 （？） 数据集 权重 异常比例
    config.abnormal_score_func = ['L2', 'L2']
    config.grad_weighted_rec_loss = True
    config.model = "mae_vit"
    config.input_size = (160, 320)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = "train"
    config.resume = False
    config.seed = 42
    config.min_data_proportion = 0.08  # 最小数据比例阈值
    config.fed_scene_dim = 64
    
   # 联邦学习参数
    config.fed_global_rounds = 50
    config.fed_local_epochs = 5
    config.fed_num_clients = 10
    config.fed_mid_dim = 256
    config.fed_scale = 1.5
    config.fed_vanilla = '-'
    config.fed_prefix_depth = 3
    config.fed_local_parts = ['adapter']
    
    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "shanghai"
    config.shanghai_path = "/mnt/private/xwy/aed-mae/data/shanghaitech_AED"
    config.shanghai_gt_path = "/mnt/private/xwy/aed-mae/data/shanghaitech_AED/Shanghai_gt"
    config.percent_abnormal = 0.3
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 5
    config.pin_mem = False

    return config
