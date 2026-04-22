import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from collections import Counter

def get_mnist_dataset(data_dir='./data'):
    """下载并返回 MNIST 训练集和测试集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 注意这里 download=True，第一次运行会自动下载
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

def partition_data(dataset, num_clients, is_iid=True, equal_size=True, alpha=0.5, seed=42):
    """
    灵活的数据集划分函数
    :param dataset: PyTorch Dataset 对象
    :param num_clients: 客户端总数
    :param is_iid: 是否独立同分布
    :param equal_size: 是否等量划分 (主要针对 IID)
    :param alpha: Dirichlet 分布参数 (仅在 Non-IID 时生效，越小越 Non-IID)
    :param seed: 随机种子，确保多台 Client 运行时拿到一致的划分结果
    :return: dict {client_id: list_of_indices}
    """
    np.random.seed(seed)
    num_samples = len(dataset)
    client_indices = {i: [] for i in range(num_clients)}

    if is_iid:
        # --- IID (独立同分布) 划分 ---
        all_idxs = np.random.permutation(num_samples)
        if equal_size:
            # 1. IID 且 等量
            splits = np.array_split(all_idxs, num_clients)
            for i in range(num_clients):
                client_indices[i] = splits[i].tolist()
        else:
            # 2. IID 且 不等量 (用大 alpha 值的 Dirichlet 生成不均匀的切割比例)
            proportions = np.random.dirichlet(np.ones(num_clients) * 2.0)
            proportions = np.round(proportions * num_samples).astype(int)
            # 修正四舍五入带来的误差
            proportions[-1] = num_samples - sum(proportions[:-1])
            
            current_idx = 0
            for i in range(num_clients):
                client_indices[i] = all_idxs[current_idx : current_idx + proportions[i]].tolist()
                current_idx += proportions[i]
    else:
        # --- Non-IID (非独立同分布) 划分 - 纯净版 Dirichlet ---
        labels = np.array(dataset.targets)
        min_size = 0 
        
        # 确保哪怕 alpha 很小，每个客户端也能分到一点数据，防止客户端空载报错
        while min_size < 10:
            client_indices = {i: [] for i in range(num_clients)}
            for k in range(10): # MNIST 有 10 个类别 (0-9)
                # 找到属于类别 k 的所有样本索引
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                
                # 直接使用 Dirichlet 分布生成比例 (取消了之前有 Bug 的过滤条件)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                
                # 将比例转化为实际的索引切分点
                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                # 按照切分点分配数据
                splits = np.split(idx_k, split_points)
                for i in range(num_clients):
                    client_indices[i] += splits[i].tolist()
            
            # 检查分得最少的客户端拿到了多少数据
            min_size = min([len(indices) for indices in client_indices.values()])

    return client_indices

def check_data_distribution(dataset, client_indices):
    """打印每个客户端的数据分布情况，用于 Debug"""
    labels = np.array(dataset.targets)
    for client_id, indices in client_indices.items():
        client_labels = labels[indices]
        label_counts = Counter(client_labels)
        total_samples = len(client_labels)
        
        # 格式化打印：显示客户端 ID，总样本数，以及 0-9 每个数字的数量
        dist_str = " | ".join([f"{k}:{label_counts.get(k, 0):4d}" for k in range(10)])
        print(f"Client {client_id:2d} (Total: {total_samples:5d}) => {dist_str}")