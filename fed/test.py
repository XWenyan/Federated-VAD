import argparse
import sys
import os

# 添加当前目录到Python路径，确保可以导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.train_dataset import AbnormalDatasetGradientsTrain
from split import FederatedDataSplitter
from torch.utils.data import DataLoader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='数据集分割')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='avenue', 
                       choices=['avenue', 'shanghai'], help='数据集名称')
    parser.add_argument('--avenue_path', type=str, default='/mnt/private/xwy/aed-mae/data/avenue', 
                       help='Avenue数据集路径')
    parser.add_argument('--shanghai_path', type=str, default='./data/shanghai', 
                       help='Shanghai数据集路径')
    parser.add_argument('--input_size', type=tuple, default=(256, 256), 
                       help='输入图像尺寸')
    parser.add_argument('--percent_abnormal', type=float, default=0.5, 
                       help='异常样本百分比')
    parser.add_argument('--input_3d', type=str, default= 'True',
                       help='是否使用3D输入')
    
    # 联邦学习参数
    parser.add_argument('--client_num', type=int, default=5, 
                       help='客户端数量')
    parser.add_argument('--split_method', type=str, default='non_iid', 
                       choices=['iid', 'non_iid'], help='数据分割方法')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='批处理大小')
    parser.add_argument('--random_seed', type=int, default=42, 
                       help='随机种子')
    
    return parser.parse_args()


def create_federated_datasets(args):
    """
    创建联邦学习数据集
    
    Args:
        args: 命令行参数
    
    Returns:
        splitter: 数据分配器实例
        client_datasets: 客户端数据集字典
    """
    print("正在加载原始数据集...")
    
    # 创建原始数据集
    original_dataset = AbnormalDatasetGradientsTrain(args)
    
    print(f"原始数据集加载完成，总样本数: {len(original_dataset)}")
    
    # 创建数据分配器
    splitter = FederatedDataSplitter(
        original_dataset, 
        args.client_num, 
        random_seed=args.random_seed
    )
    
    # 分割数据
    print(f"正在使用 {args.split_method} 方法分割数据到 {args.client_num} 个客户端...")
    splitter.split_data(split_method=args.split_method, alpha=0.5)
    
    # 获取所有客户端数据集
    client_datasets = splitter.get_all_client_datasets()
    
    # 打印统计信息
    splitter.print_client_statistics()
    
    return splitter, client_datasets


def create_client_dataloaders(client_datasets, batch_size=32):
    """
    为每个客户端创建数据加载器
    
    Args:
        client_datasets: 客户端数据集字典
        batch_size: 批处理大小
    
    Returns:
        client_dataloaders: 客户端数据加载器字典
    """
    client_dataloaders = {}
    
    for client_id, dataset in client_datasets.items():
        client_dataloaders[client_id] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
    print(f"已为 {len(client_dataloaders)} 个客户端创建数据加载器")
    return client_dataloaders


def test_dataloaders(client_dataloaders):
    """
    测试数据加载器
    
    Args:
        client_dataloaders: 客户端数据加载器字典
    """
    print("\n测试数据加载器...")
    
    for client_id, dataloader in client_dataloaders.items():
        print(f"\n客户端 {client_id}:")
        print(f"  批次数: {len(dataloader)}")
        
        # 获取一个批次的数据
        for batch_idx, (imgs, gradients, targets) in enumerate(dataloader):
            print(f"  批次 {batch_idx}:")
            print(f"    输入图像: {imgs.shape}")
            print(f"    梯度图像: {gradients.shape}")
            print(f"    目标图像: {targets.shape}")
            
            # 只测试第一个批次
            if batch_idx == 0:
                break


def main():
    """主函数"""
    args = parse_args()
    
    try:
        # 创建联邦学习数据集
        splitter, client_datasets = create_federated_datasets(args)
        
        # 创建数据加载器
        client_dataloaders = create_client_dataloaders(
            client_datasets, 
            batch_size=args.batch_size
        )
        
        # 测试数据加载器（可选）
        test_dataloaders(client_dataloaders)
        
        print("\n数据分配完成")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()