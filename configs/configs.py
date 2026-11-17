import ml_collections


def get_configs_avenue():
    config = ml_collections.ConfigDict()
    config.batch_size = 50
    config.epochs = 140
    config.mask_ratio = 0.5
    config.start_TS_epoch = 100
    config.masking_method = "random_masking"
    config.output_dir = "experiments/avenue"
    config.abnormal_score_func = ['L2', 'L2']
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (320, 640)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = 'train' # 或train 测试和训练
    config.resume = False
    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "avenue"
    config.avenue_path = "/mnt/private/xwy/aed-mae/data/avenue"
    config.avenue_gt_path = "/mnt/private/xwy/aed-mae/data/avenue/Avenue_gt"
    config.percent_abnormal = 0.25
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 10
    config.pin_mem = False

    return config


def get_configs_shanghai():
    config = ml_collections.ConfigDict()
    config.batch_size = 50
    config.epochs = 200
    config.mask_ratio = 0.5
    config.start_TS_epoch = 100
    config.masking_method = "random_masking"
    config.output_dir = "experiments/shanghai" # the checkpoints will be loaded from here
    config.abnormal_score_func = 'L1'
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (160, 320)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = "inference"
    config.resume=False

    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "shanghai"
    config.shanghai_path = "/media/alin/hdd/SanhaiTech"
    config.shanghai_gt_path = "/media/alin/hdd/Transformer_Labels/Shanghai_gt"
    config.percent_abnormal = 0.25
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 10
    config.pin_mem = False

    return config

def get_configs_ped2():
    config = ml_collections.ConfigDict()

    # Training
    config.batch_size = 100
    config.epochs = 300
    config.mask_ratio = 0.75 # 0.5 0.75最佳
    config.start_TS_epoch = 300
    config.masking_method = "random_masking"
    config.output_dir = "experiments/ped2"
    config.abnormal_score_func = ['L2', 'L2']
    config.grad_weighted_rec_loss = True
    config.model = "mae_vit"
    config.input_size = (240, 360)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = 'train'
    config.resume = False

    # Optimizer
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset
    config.dataset = "ped2"
    config.ped2_path = "/mnt/private/xwy/aed-mae/data/ped2"
    config.ped2_gt_path = "/mnt/private/xwy/aed-mae/data/ped2/ped2_gt"
    config.percent_abnormal = 0.25
    config.input_3d = True
    config.device = "cuda"

    # System
    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 10
    config.pin_mem = False

    return config
