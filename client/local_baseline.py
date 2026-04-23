import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

# 导入共享的模型结构
from model import SimpleCNN

def run_local_baseline(client_id, data_path, epochs=5, val_ratio=0.2):
    """
    运行单机本地基线测试
    :param client_id: 客户端标识
    :param data_path: 本地数据集路径
    :param epochs: 训练轮数
    :param val_ratio: 验证集比例 (默认 20%)
    """
    print(f"========== 开始测试客户端 {client_id} 的本地极限界限 ==========")
    
    # 1. 加载本地数据
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据集文件：{data_path}。请先运行 prepare_data.py")
        
    full_dataset = torch.load(data_path)
    total_samples = len(full_dataset)
    
    # 2. 划分训练集和验证集 (核心步骤)
    val_size = int(total_samples * val_ratio)
    train_size = total_samples - val_size
    
    # 使用随机种子确保每次划分一致 (可选)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    print(f"总数据量: {total_samples} | 划分 -> 训练集: {train_size}, 验证集: {val_size}")
    
    # 3. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 4. 初始化模型和优化器
    model = SimpleCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    # 5. 开始独立训练循环
    for epoch in range(1, epochs + 1):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100. * correct / len(val_loader.dataset)
        
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}%")
              
    print("========== 本地基线测试完成 ==========\n")
    # ==========================================
    # 终极审判：在全局测试集上评估这个“井底之蛙”模型
    # ==========================================
    
    global_test_path = "./dist_data/server/test_data.pt" # 预设的挂载点
    
    if not os.path.exists(global_test_path):
        print(f"⚠️ 警告：未检测到全局测试集挂载点 {global_test_path}")
        print("请确保执行 docker run 时映射了 server 的数据目录。")
        return model

    print(f"🌍 正在加载挂载的全局测试集进行大考...")
    # 直接加载 prepare_data.py 生成的 test_dataset 对象
    global_test_dataset = torch.load(global_test_path, weights_only=False)
    global_test_loader = DataLoader(global_test_dataset, batch_size=1000, shuffle=False)
    
    model.eval() # 开启评估模式
    global_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in global_test_loader:
            output = model(data)
            global_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    global_accuracy = 100. * correct / len(global_test_loader.dataset)
    print(f"🚨 【残酷真相】 {client_id} 模型全局准确率: {correct}/10000 ({global_accuracy:.2f}%)")
    print(f"对比：联邦学习 (FedAvg) 的全局准确率通常在 90% 以上。\n")
    
    return model

if __name__ == "__main__":

    current_client_id = "client_3"
    # 【核心修改】适配 Docker 容器环境
    # 在容器内，无论哪个客户端，它的专属数据都被映射到了这个固定路径
    container_data_path = f"./dist_data/{current_client_id}/local_data.pt"
    
    # 获取当前容器的身份标识
        
    if os.path.exists(container_data_path):
        run_local_baseline(current_client_id, container_data_path, epochs=50)
    else:
        print(f"❌ 找不到数据集文件：{container_data_path}")
        print("请检查 docker-compose.yml 中的 volumes 挂载是否正确。")




