import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from shared.Lora_model import SimpleCNN, count_trainable_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 当前使用的计算设备: {device}")
# 配置参数
CLIENT_DATA_PATH = "./dist_data/client_0/local_data.pt"
GLOBAL_TEST_PATH = "./dist_data/server/test_data.pt"
EPOCHS = 10
LEARNING_RATE = 0.01

def train_and_evaluate(model_name, model, train_loader, test_loader):
    print(f"\n========== 开始训练: {model_name} ==========")
    trainable_params = count_trainable_parameters(model)
    print(f"[{model_name}] 需训练/加密的参数量: {trainable_params:,}")

    model = model.to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, momentum=0.5)
    
    accuracy_history = []
    
    for epoch in range(1, EPOCHS + 1):
        # 1. 训练阶段
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
        # 2. 全局测试集评估阶段
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        acc = 100. * correct / len(test_loader.dataset)
        accuracy_history.append(acc)
        print(f"Epoch {epoch}/{EPOCHS} | 全局测试集准确率: {acc:.2f}%")
    model = model.cpu()
    return accuracy_history, trainable_params

if __name__ == "__main__":
    if not os.path.exists(CLIENT_DATA_PATH) or not os.path.exists(GLOBAL_TEST_PATH):
        print("❌ 找不到数据文件，请先运行 prepare_data.py！")
        exit(1)

    print("📦 正在加载数据...")
    train_dataset = torch.load(CLIENT_DATA_PATH, weights_only=False)
    test_dataset = torch.load(GLOBAL_TEST_PATH, weights_only=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 实验 1：全量微调 Baseline (不使用 LoRA)
    model_base = SimpleCNN(use_lora=False)
    acc_base, params_base = train_and_evaluate("全量微调 Baseline", model_base, train_loader, test_loader)

    # 实验 2：LoRA (Rank = 4)
    model_lora_r4 = SimpleCNN(use_lora=True, rank=4)
    acc_lora_r4, params_r4 = train_and_evaluate("LoRA (Rank=4)", model_lora_r4, train_loader, test_loader)

    # 实验 3：LoRA (Rank = 16)
    model_lora_r16 = SimpleCNN(use_lora=True, rank=16)
    acc_lora_r16, params_r16 = train_and_evaluate("LoRA (Rank=16)", model_lora_r16, train_loader, test_loader)

    # ================= 绘制对比曲线 =================
    print("\n📊 正在生成对比曲线图...")
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, EPOCHS + 1)
    
    plt.plot(epochs_range, acc_base, marker='o', linestyle='-', linewidth=2, label=f'Baseline (Params: {params_base:,})')
    plt.plot(epochs_range, acc_lora_r16, marker='s', linestyle='--', linewidth=2, label=f'LoRA r=16 (Params: {params_r16:,})')
    plt.plot(epochs_range, acc_lora_r4, marker='^', linestyle='-.', linewidth=2, label=f'LoRA r=4 (Params: {params_r4:,})')
    
    plt.title('Learning Curve: Full Fine-tuning vs LoRA', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Global Test Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    
    # 保存图片
    plt.savefig('lora_comparison_curve.png', dpi=300, bbox_inches='tight')
    print("✅ 曲线图已保存为: lora_comparison_curve.png")