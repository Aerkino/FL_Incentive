import torch
import torch.optim as optim
from torchvision import datasets, transforms
from shared.model import SimpleCNN 

def pretrain_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 统一定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 下载并加载训练集和测试集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print("正在为联邦学习准备 '预训练基座'...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = torch.nn.functional.nll_loss(model(data), target)
        loss.backward()
        optimizer.step()
        
        # 仅训练 200 个 batch (约 12800 个样本)，赋予基础视觉特征
        if batch_idx >= 10: 
            break
            
    print("✅ 预训练阶段结束，正在评估基座模型基线准确率...")
    
    # ==========================================
    # 新增：测试集评估逻辑
    # ==========================================
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f"\n⭐ [基座模型 Baseline] 测试集 Loss: {test_loss:.4f} | 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%) ⭐\n")

    # 保存预训练权重
    torch.save(model.state_dict(), 'pretrained_base.pth')
    print("✅ 预训练基座已保存为 pretrained_base.pth，可以开始联邦微调了！")

if __name__ == '__main__':
    pretrain_and_evaluate()