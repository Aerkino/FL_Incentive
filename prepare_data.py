# import os
# import torch
# from data_utils import get_mnist_dataset, partition_data, check_data_distribution

# def generate_client_data(num_clients=3, is_iid=True, alpha=0.5):
#     # 拿到训练集和测试集
#     train_dataset, test_dataset = get_mnist_dataset()
    
#     # ==========================================
#     # 新增：保存全局测试集给 Server
#     # ==========================================
#     server_dir = "./dist_data/server"
#     os.makedirs(server_dir, exist_ok=True)
#     # 对于测试集，我们不需要切分，直接整体保存即可
#     torch.save(test_dataset, f"{server_dir}/test_data.pt")
#     print(f"✅ 全局测试集已就绪 (发给 Server)：{len(test_dataset)} 样本")

#     # 获取划分索引
#     indices_dict = partition_data(train_dataset, num_clients, is_iid=is_iid, alpha=alpha)
    
#     print("\n=== 本次划分的数据分布情况 (0-9 标签数量) ===")
#     check_data_distribution(train_dataset, indices_dict)
#     print("==============================================\n")

#     for client_id, indices in indices_dict.items():
#         # 创建该客户端的独立文件夹
#         client_dir = f"./dist_data/client_{client_id}"
#         os.makedirs(client_dir, exist_ok=True)
        
#         # 提取该客户端的数据子集并保存为独立文件
#         # 这样容器里只需要 torch.load() 这个文件，不需要原始 MNIST 结构
#         subset = torch.utils.data.Subset(train_dataset, indices)
#         data_to_save = []
#         for img, label in subset:
#             data_to_save.append((img, label))
            
#         torch.save(data_to_save, f"{client_dir}/local_data.pt")
#         print(f"客户端 {client_id} 数据已就绪：{len(data_to_save)} 样本")

# if __name__ == "__main__":
#     generate_client_data(num_clients=5, is_iid=False, alpha=0.5)


import os
import torch
import random  # 新增：用于随机抽样
from data_utils import get_mnist_dataset, partition_data, check_data_distribution
from collections import Counter


def generate_client_data(num_clients=3, is_iid=True, alpha=0.5, max_samples_per_client=100):
    # 拿到训练集和测试集
    train_dataset, test_dataset = get_mnist_dataset()
    
    # ==========================================
    # 保存全局测试集给 Server
    # ==========================================
    server_dir = "./dist_data/server"
    os.makedirs(server_dir, exist_ok=True)
    # 对于测试集，我们不需要切分，直接整体保存即可
    torch.save(test_dataset, f"{server_dir}/test_data.pt")
    print(f"✅ 全局测试集已就绪 (发给 Server)：{len(test_dataset)} 样本")

    # 获取划分索引 (这里依然是原始的庞大划分)
    indices_dict = partition_data(train_dataset, num_clients, is_iid=is_iid, alpha=alpha)
    
    print("\n=== 原始分配的数据分布情况 (截断前) ===")
    check_data_distribution(train_dataset, indices_dict)
    print("=======================================\n")

    for client_id, indices in indices_dict.items():
        # 创建该客户端的独立文件夹
        client_dir = f"./dist_data/client_{client_id}"
        os.makedirs(client_dir, exist_ok=True)
        
        # 提取该客户端的原始数据子集
        subset = torch.utils.data.Subset(train_dataset, indices)
        original_len = len(subset)
        
        # ==========================================
        # 🚨 核心修改：暴力抽样，人为制造数据稀缺
        # ==========================================
        if original_len > max_samples_per_client:
            # 在当前 subset 的长度范围内，随机抽取 max_samples_per_client 个索引
            sampled_indices = random.sample(range(original_len), max_samples_per_client)
            # 嵌套 Subset，实现数据的二次截断
            subset = torch.utils.data.Subset(subset, sampled_indices)
        
        # 将截断后的数据转存为 List 方便容器内直接 load
        data_to_save = []
        for img, label in subset:
            data_to_save.append((img, label))
            
        torch.save(data_to_save, f"{client_dir}/local_data.pt")
        print(f"✅ 客户端 {client_id} 数据已就绪：{len(data_to_save)} 样本 (原分配: {original_len} 样本，已执行灾难性截断)")


def generate_extreme_client_data():
    num_clients = 5
    samples_per_class = 100 # 每种数字只给 50 张，每个客户端总共 100 张
    
    # 1. 拿到训练集和测试集
    train_dataset, test_dataset = get_mnist_dataset()
    
    # 2. 保存全局测试集给 Server
    server_dir = "./dist_data/server"
    os.makedirs(server_dir, exist_ok=True)
    torch.save(test_dataset, f"{server_dir}/test_data.pt")
    print(f"✅ 全局测试集已就绪 (发给 Server)：{len(test_dataset)} 样本")

    # 3. 将所有训练集数据按标签 (0-9) 分门别类提取出来
    # 这样我们就能精准地从中抽牌
    print("\n📦 正在按标签整理原始数据...")
    class_indices = {i: [] for i in range(10)}
    for idx, (img, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
        
    print("\n=== 开始进行极端的病态划分 (Pathological Non-IID) ===")
    
    # 4. 强制物理隔离：每个客户端只分配 2 种连续的标签
    for client_id in range(num_clients):
        client_dir = f"./dist_data/client_{client_id}"
        os.makedirs(client_dir, exist_ok=True)
        
        # 客户端 0 拿 [0, 1]，客户端 1 拿 [2, 3]...
        label_1 = client_id * 2
        label_2 = client_id * 2 + 1
        
        # 从这两个类别中，分别随机抽取 samples_per_class (50) 张图片的索引
        idx_label_1 = random.sample(class_indices[label_1], samples_per_class)
        idx_label_2 = random.sample(class_indices[label_2], samples_per_class)
        
        # 合并索引，提取真实数据
        final_indices = idx_label_1 + idx_label_2
        # 打乱顺序，防止训练时先吃完全部 0 再吃 1
        random.shuffle(final_indices) 
        
        subset = torch.utils.data.Subset(train_dataset, final_indices)
        
        # 保存为 List
        data_to_save = []
        for img, label in subset:
            data_to_save.append((img, label))
            
        torch.save(data_to_save, f"{client_dir}/local_data.pt")
        print(f"🚨 客户端 {client_id} 数据就绪 | 样本数: {len(data_to_save)} | 绝对隔离标签: [{label_1}, {label_2}]")

    print("\n=======================================================")
    print("😈 灾难现场布置完毕！请重新运行你的联邦学习测试代码。")

def check_truncated_distribution(num_clients=5):
    print("\n" + "="*15 + " 截断后的真实数据分布 (盘点物理文件) " + "="*15)
    print(f"{'Client ID':<10} | {'总样本量':<8} | {'标签分布 (类别: 数量)':<40}")
    print("-" * 75)
    
    for i in range(num_clients):
        file_path = f"./dist_data/client_{i}/local_data.pt"
        if not os.path.exists(file_path):
            print(f"❌ 找不到客户端 {i} 的数据文件：{file_path}")
            continue
            
        # 直接把物理文件加载到内存
        data = torch.load(file_path, weights_only=False)
        
        # 提取所有的 label
        labels = [label for img, label in data]
        total_samples = len(labels)
        
        # 使用 Counter 统计每个数字(0-9)出现的次数
        label_counts = Counter(labels)
        
        # 按照标签 0-9 排序，并格式化输出
        sorted_counts = sorted(label_counts.items())
        dist_str = "  ".join([f"[{k}]:{v}" for k, v in sorted_counts])
        
        print(f"Client_{i:<3} | {total_samples:<8} | {dist_str}")
        
    print("-" * 75 + "\n")


def generate_perfect_iid_data():
    num_clients = 5
    samples_per_class = 20  # 每个标签 20 张 * 10 个标签 = 200 张/客户端
    
    # 1. 拿到训练集和测试集
    train_dataset, test_dataset = get_mnist_dataset()
    
    # 2. 保存全局测试集给 Server
    server_dir = "./dist_data/server"
    os.makedirs(server_dir, exist_ok=True)
    torch.save(test_dataset, f"{server_dir}/test_data.pt")
    print(f"✅ 全局测试集已就绪 (发给 Server)：{len(test_dataset)} 样本")

    # 3. 将所有训练集数据按标签 (0-9) 整理到不同的桶里
    print("\n📦 正在按标签整理原始数据...")
    class_indices = {i: [] for i in range(10)}
    for idx, (img, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
        
    print("\n=== 开始进行完美 IID 均匀分配 (对照组 Baseline) ===")
    
    # 4. 绝对平均分配：每个客户端从 0-9 的每个桶里精准捞取 20 张
    for client_id in range(num_clients):
        client_dir = f"./dist_data/client_{client_id}"
        os.makedirs(client_dir, exist_ok=True)
        
        final_indices = []
        
        # 遍历 0 到 9 的每个数字标签
        for label in range(10):
            # 从该标签的桶里随机抽出 20 张图片的索引
            sampled_idx = random.sample(class_indices[label], samples_per_class)
            final_indices.extend(sampled_idx)
            
            # 【关键】为了确保不同客户端拿到的数据不重复，把抽过的索引从桶里删掉
            for idx in sampled_idx:
                class_indices[label].remove(idx)
                
        # 此时 final_indices 里刚好有 200 个索引。
        # 必须打乱它们！否则客户端训练时会先连续吃 20 个 0，再吃 20 个 1... 导致内部梯度震荡
        random.shuffle(final_indices)
        
        subset = torch.utils.data.Subset(train_dataset, final_indices)
        
        # 保存为 List，方便容器读取
        data_to_save = []
        for img, label in subset:
            data_to_save.append((img, label))
            
        torch.save(data_to_save, f"{client_dir}/local_data.pt")
        print(f"🌟 客户端 {client_id} 数据就绪 | 总样本: {len(data_to_save)} | 分布: 0-9 各 {samples_per_class} 张 (完美平衡)")

    print("\n=======================================================")
    print("⚖️ IID 对照组数据生成完毕！请运行 FL 测试，见证准确率回归。")

if __name__ == "__main__":
    # 开启 Non-IID (alpha=0.5) 并且限制每个客户端最多只有 100 张图片！
    # generate_client_data(num_clients=5, is_iid=False, alpha=0.01, max_samples_per_client=200)
    # generate_extreme_client_data()
    generate_perfect_iid_data()
    # 生成完数据后，直接盘点一下物理文件里的真实分布情况
    check_truncated_distribution(num_clients=5)