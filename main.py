import argparse
import datetime
import json
import os
import time
from pathlib import Path

from timm.optim import optim_factory
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter

from configs.configs import get_configs_avenue, get_configs_shanghai, get_configs_ped2
from data.test_dataset import AbnormalDatasetGradientsTest
from data.train_dataset import AbnormalDatasetGradientsTrain
from engine_train import train_one_epoch, test_one_epoch
from inference import inference
from model.model_factory import mae_cvt_patch16, mae_cvt_patch8
from util import misc
import torch



def main(args):
    # 打印当前工作目录（脚本所在目录）
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # 打印命令行参数（格式化显示）
    print("{}".format(args).replace(', ', ',\n'))
    # 初始化TensorBoard日志写入器，日志保存到输出目录
    log_writer = SummaryWriter(log_dir=args.output_dir)

    device = args.device
    # 如果运行类型为训练，则加载训练数据集
    if args.run_type =='train':
        dataset_train = AbnormalDatasetGradientsTrain(args) # 初始化训练数据集
        print(dataset_train) # 打印数据集信息
        sampler_train = torch.utils.data.RandomSampler(dataset_train) # 创建训练数据的随机采样器
        # 创建训练数据加载器（批量加载数据）
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    # 加载测试数据集（无论训练还是推理都需要测试集）
    dataset_test = AbnormalDatasetGradientsTest(args)
    print(dataset_test) # 打印测试数据集信息
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # define the model
    # 根据数据集类型定义模型（Avenue用patch16，其他用patch8）
    if args.dataset == 'avenue':
        model = mae_cvt_patch16(norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                                                use_only_masked_tokens_ab=args.use_only_masked_tokens_ab, # 是否仅使用掩码token计算异常分数
                                                abnormal_score_func=args.abnormal_score_func, # 异常分数计算函数
                                                masking_method=args.masking_method, # 掩码方法
                                                grad_weighted_loss=args.grad_weighted_rec_loss).float() # 是否使用梯度加权重建损失
    else:
        model = mae_cvt_patch8(args, norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                                                use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                                                abnormal_score_func=args.abnormal_score_func,
                                                masking_method=args.masking_method,
                                                grad_weighted_loss=args.grad_weighted_rec_loss).float()
    # model = build_prefix(model, args) 
    model.to(device)
    if args.run_type == "train":
        do_training(args, data_loader_test, data_loader_train, device, log_writer, model)
    elif args.run_type == "inference":
        student = torch.load(args.output_dir + "/checkpoint-best-student.pth", weights_only=False)['model']
        teacher = torch.load(args.output_dir + "/checkpoint-best.pth", weights_only=False)['model']
        for key in student:
            if 'student' in key:
                teacher[key] = student[key]
        model.load_state_dict(teacher, strict=False)
        with torch.no_grad():
            inference(model, data_loader_test, device, args=args)



def do_training(args, data_loader_test, data_loader_train, device, log_writer, model):
    print("actual lr: %.2e" % args.lr)
    # following timm: set wd as 0 for bias and norm layers
    # 按timm库的方式划分参数组（对偏置和归一化层设置权重衰减为0）
    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler() # 初始化混合精度训练的损失缩放器
    misc.load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler) # 加载已保存的模型状态（如需断点续训）
    # 打印训练总轮次信息
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_micro = 0.0 # 记录最佳的micro指标（用于保存最佳模型）
    best_micro_student = 0.0 # 记录学生模型的最佳micro指标
    for epoch in range(args.start_epoch, args.epochs):
        # 训练一个epoch并返回训练统计信息
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch,
            log_writer=log_writer,
            args=args
        )
        # 格式化训练日志（添加'train_'前缀）
        log_stats_train = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        # 测试一个epoch并返回测试统计信息
        test_stats = test_one_epoch(
            model, data_loader_test, device, epoch, log_writer=log_writer, args=args
        )
        log_stats_test = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}
        # 如果设置了输出目录，保存最新模型
        if args.output_dir:
            misc.save_model(args=args, model=model, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, latest=True)
        # 如果当前测试的micro指标优于历史最佳，更新最佳指标并保存最佳模型
        if test_stats['micro'] > best_micro:
            best_micro = test_stats['micro']
            misc.save_model(args=args, model=model, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, best=True)
        # 如果当前epoch >= 开始训练学生模型的epoch，判断是否保存最佳学生模型
        if args.start_TS_epoch <= epoch:
            if test_stats['micro'] > best_micro_student:
                best_micro_student = test_stats['micro']
                misc.save_model(args=args, model=model, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, best=True, student=True)
        # 如果设置了输出目录，写入日志
        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log_train.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_train) + "\n")
            with open(os.path.join(args.output_dir, "log_test.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_test) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ped2')
    args = parser.parse_args()
    if args.dataset == 'avenue':
        args = get_configs_avenue()
    elif args.dataset == 'shanghai':
        args = get_configs_shanghai()
    elif args.dataset == 'ped2':
        args = get_configs_ped2()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
