import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, seed=42):
        super(LoRALinear, self).__init__()
        
        # 1. 基础线性层 (冻结)
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        # 2. 矩阵 A (解冻：现在它是可训练的)
        torch.manual_seed(seed)
        self.lora_A = nn.Parameter(torch.randn(in_features, r) / math.sqrt(in_features))
        self.lora_A.requires_grad = True # 修改：允许 A 更新

        # 3. 矩阵 B (可训练)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.lora_B.requires_grad = True 

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

class SimpleCNNLora(nn.Module):
    def __init__(self, r=8, lora_alpha=16, seed=42):
        super(SimpleCNNLora, self).__init__()
        torch.manual_seed(seed)
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1_temp = nn.Linear(320, 50) 
        self.fc2_temp = nn.Linear(50, 10)

        # 加载预训练基座
        base_path = os.path.join(os.path.dirname(__file__), '..', 'pretrained_base.pth')
        if os.path.exists(base_path):
            self.load_state_dict(torch.load(base_path, map_location='cpu'), strict=False)

        # 替换为 LoRA 层并拷贝权重
        self.fc1 = LoRALinear(320, 50, r=r, lora_alpha=lora_alpha, seed=seed)
        self.fc1.linear.weight.data = self.fc1_temp.weight.data.clone()
        self.fc1.linear.bias.data = self.fc1_temp.bias.data.clone()
        
        self.fc2 = LoRALinear(50, 10, r=r, lora_alpha=lora_alpha, seed=seed)
        self.fc2.linear.weight.data = self.fc2_temp.weight.data.clone()
        self.fc2.linear.bias.data = self.fc2_temp.bias.data.clone()

        # 冻结卷积
        for param in self.conv1.parameters(): param.requires_grad = False
        for param in self.conv2.parameters(): param.requires_grad = False

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)