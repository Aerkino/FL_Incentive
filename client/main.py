import grpc
import time
import os
import random
import io

# 引入 PyTorch 生态
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入你自己的模型
from model import SimpleCNN

# 导入编译生成的代码 (注意服务名从 Sum 变成了 Learning)
import fl_pb2
import fl_pb2_grpc

# 配置参数
CLIENT_ID = os.environ.get("CLIENT_ID", f"Client-{random.randint(100,999)}")
SERVER_ADDR = os.environ.get("SERVER_ADDR", "server:50051")
TOTAL_ROUNDS = int(os.environ.get("TOTAL_ROUNDS", "5"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "2")) # 预留本地训练 Epoch 数
LEARNING_RATE = 0.005

# --- 核心工具函数 ---

def load_local_data():
    """从 Docker 数据卷加载专属数据集"""
    data_path = "/app/data/local_data.pt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[{CLIENT_ID}] 未找到挂载数据，请检查 docker-compose volume 配置: {data_path}")
    
    local_data = torch.load(data_path,weights_only=False)
    # 包装成 DataLoader，方便分批次训练
    return DataLoader(local_data, batch_size=32, shuffle=True)

def serialize_weights(state_dict):
    """将 PyTorch 模型参数无硬盘序列化为 bytes"""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()

def deserialize_weights(serialized_bytes):
    """将 bytes 反序列化为 PyTorch 模型参数"""
    buffer = io.BytesIO(serialized_bytes)
    return torch.load(buffer)

# --- 训练逻辑 ---

def local_train(model, train_loader):
    """标准的 PyTorch 本地训练循环"""
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.2)
    model.train()
    
    print(f"[{CLIENT_ID}] 开始本地训练 (Epochs: {LOCAL_EPOCHS}, 样本数: {len(train_loader.dataset)})...")
    
    for epoch in range(LOCAL_EPOCHS):
        epoch_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"[{CLIENT_ID}] \t Epoch {epoch+1}/{LOCAL_EPOCHS} 平均 Loss: {epoch_loss/len(train_loader):.4f}")
        
    return model.state_dict(), len(train_loader.dataset)

# --- 主控循环 ---

def run():
    print(f"[{CLIENT_ID}] 正在初始化并连接到服务器 {SERVER_ADDR} ...")
    
    # 1. 准备本地数据和模型结构
    train_loader = load_local_data()
    model = SimpleCNN()
    
    with grpc.insecure_channel(SERVER_ADDR) as channel:
        stub = fl_pb2_grpc.FederatedLearningStub(channel)
        
        for current_round in range(1, TOTAL_ROUNDS + 1):
            print(f"\n[{CLIENT_ID}] ================= 开始执行第 {current_round} 轮 =================")
            
            # 2. 同步等待：轮询获取本轮的全局模型
            print(f"[{CLIENT_ID}] 等待服务器下发第 {current_round} 轮的全局模型...")
            while True:
                check_req = fl_pb2.GlobalModelRequest(round_number=current_round)
                check_resp = stub.GetGlobalModel(check_req)
                
                if check_resp.is_ready:
                    # 获取成功，反序列化并加载到本地
                    global_state_dict = deserialize_weights(check_resp.global_weights)
                    model.load_state_dict(global_state_dict)
                    print(f"[{CLIENT_ID}] 成功加载全局模型！")
                    break
                
                time.sleep(1) # 没准备好就接着睡
                
            # 3. 执行本地深度学习训练
            local_weights, num_samples = local_train(model, train_loader)
            
            # 4. 序列化并上传本地模型
            print(f"[{CLIENT_ID}] 序列化参数并上传中...")
            upload_req = fl_pb2.LocalModel(
                client_id=CLIENT_ID,
                model_weights=serialize_weights(local_weights),
                num_samples=num_samples,
                round_number=current_round
            )
            response = stub.UploadLocalModel(upload_req)
            print(f"[{CLIENT_ID}] 上传状态: {response.message}")
            
            if not response.success:
                print(f"[{CLIENT_ID}] ⚠️ 上传失败，可能是轮次错位，跳出循环。")
                break
                
        print(f"\n[{CLIENT_ID}] 恭喜！完成所有 {TOTAL_ROUNDS} 轮训练，客户端安全退出。")

if __name__ == '__main__':
    run()