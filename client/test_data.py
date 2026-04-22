from data_utils import get_mnist_dataset, partition_data, check_data_distribution

# 1. 下载并加载数据集
print("正在加载 MNIST...")
dataset = get_mnist_dataset()

# 2. 测试 IID 等量划分
print("\n=== IID 等量划分测试 (3 个客户端) ===")
iid_indices = partition_data(dataset, num_clients=10, is_iid=True, equal_size=True)
check_data_distribution(dataset, iid_indices)

# 3. 测试 Non-IID 划分 (alpha = 0.1，极其异构)
print("\n=== Non-IID 划分测试 (alpha=0.1, 3 个客户端) ===")
noniid_indices = partition_data(dataset, num_clients=10, is_iid=False, alpha=0.5)
check_data_distribution(dataset, noniid_indices)