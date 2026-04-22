import os
import torch
from data_utils import get_mnist_dataset, partition_data, check_data_distribution

def generate_client_data(num_clients=3, is_iid=True, alpha=0.5):
    # 拿到训练集和测试集
    train_dataset, test_dataset = get_mnist_dataset()
    
    # ==========================================
    # 新增：保存全局测试集给 Server
    # ==========================================
    server_dir = "./dist_data/server"
    os.makedirs(server_dir, exist_ok=True)
    # 对于测试集，我们不需要切分，直接整体保存即可
    torch.save(test_dataset, f"{server_dir}/test_data.pt")
    print(f"✅ 全局测试集已就绪 (发给 Server)：{len(test_dataset)} 样本")

    # 获取划分索引
    indices_dict = partition_data(train_dataset, num_clients, is_iid=is_iid, alpha=alpha)
    
    print("\n=== 本次划分的数据分布情况 (0-9 标签数量) ===")
    check_data_distribution(train_dataset, indices_dict)
    print("==============================================\n")

    for client_id, indices in indices_dict.items():
        # 创建该客户端的独立文件夹
        client_dir = f"./dist_data/client_{client_id}"
        os.makedirs(client_dir, exist_ok=True)
        
        # 提取该客户端的数据子集并保存为独立文件
        # 这样容器里只需要 torch.load() 这个文件，不需要原始 MNIST 结构
        subset = torch.utils.data.Subset(train_dataset, indices)
        data_to_save = []
        for img, label in subset:
            data_to_save.append((img, label))
            
        torch.save(data_to_save, f"{client_dir}/local_data.pt")
        print(f"客户端 {client_id} 数据已就绪：{len(data_to_save)} 样本")

if __name__ == "__main__":
    generate_client_data(num_clients=5, is_iid=False, alpha=0.8)
