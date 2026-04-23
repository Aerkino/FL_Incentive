import grpc
import time
import os
import random
import io

# 引入 PyTorch 生态
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from random_projection import BlockchainRandomProjector
# 导入你自己的模型
from model import SimpleCNN

# 导入编译生成的代码 (注意服务名从 Sum 变成了 Learning)
import fl_pb2
import fl_pb2_grpc
from smc_engine import PyTorchSMCEngine

# 配置参数
CLIENT_ID = os.environ.get("CLIENT_ID", f"Client-{random.randint(100,999)}")
SERVER_ADDR = os.environ.get("SERVER_ADDR", "server:50051")
TOTAL_ROUNDS = int(os.environ.get("TOTAL_ROUNDS", "5"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "2")) # 预留本地训练 Epoch 数
LEARNING_RATE = 0.005
ALL_CLIENT_IDS = ["client_0", "client_1", "client_2", "client_3", "client_4"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{CLIENT_ID}] 当前使用的计算设备: {device}")
# --- 核心工具函数 ---

def load_local_data():
    """从 Docker 数据卷加载专属数据集"""
    client_folder = CLIENT_ID.lower() # 变成 client_0
    data_path = f"./dist_data/{client_folder}/local_data.pt"

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
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.2)
    model.train()
    
    # 🌟 模拟投毒判定
    is_poisoner = (CLIENT_ID == "Client_4")
    if is_poisoner:
        print(f"[⚠️ ATTACK] {CLIENT_ID} 正在执行标签翻转投毒...")

    print(f"[{CLIENT_ID}] 开始本地训练 (Epochs: {LOCAL_EPOCHS}, 样本数: {len(train_loader.dataset)})...")
    
    for epoch in range(LOCAL_EPOCHS):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # 🌟 执行投毒：标签翻转 (9 - y)
            if is_poisoner:
                target = 9 - target

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"[{CLIENT_ID}] \t Epoch {epoch+1}/{LOCAL_EPOCHS} 平均 Loss: {epoch_loss/len(train_loader):.4f}")
    model = model.cpu() # 训练完成后移回 CPU，准备序列化    
    return model.state_dict(), len(train_loader.dataset)

# --- 主控循环 ---

def run():
    print(f"[{CLIENT_ID}] 正在初始化并连接到服务器 {SERVER_ADDR} ...")
    
    # 1. 准备本地数据和模型结构
    train_loader = load_local_data()
    model = SimpleCNN()
    current_blockhash = "0xABC123..." 
    projector = BlockchainRandomProjector(target_dim=128, blockhash_seed=current_blockhash)
    smc_engine = PyTorchSMCEngine(CLIENT_ID, ALL_CLIENT_IDS)

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
                
                time.sleep(0.1) # 没准备好就接着睡
            


            # 3. 执行本地深度学习训练
            local_weights, num_samples = local_train(model, train_loader)
            

            rp_vec = projector.project(local_weights)
            rp_vec_list = rp_vec.tolist()
            local_weights = smc_engine.mask_state_dict(local_weights)

            # 4. 序列化并上传本地模型
            print(f"[{CLIENT_ID}] 序列化参数并上传中...")
            upload_req = fl_pb2.LocalModel(
                client_id=CLIENT_ID,
                model_weights=serialize_weights(local_weights),
                num_samples=num_samples,
                round_number=current_round,
                rp_vector=rp_vec_list
            )
            response = stub.UploadLocalModel(upload_req)
            print(f"[{CLIENT_ID}] 上传状态: {response.message}")
            
            if not response.success:
                print(f"[{CLIENT_ID}] ⚠️ 上传失败，可能是轮次错位，跳出循环。")
                break
                
        print(f"\n[{CLIENT_ID}] 恭喜！完成所有 {TOTAL_ROUNDS} 轮训练，客户端安全退出。")

if __name__ == '__main__':
    run()


# import grpc
# import time
# import os
# import random
# import io
# import copy

# # 引入 PyTorch 生态
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # 导入你自己的模型 (假设你后续会切换到 LoraModel)
# from model import SimpleCNN 
# # 如果要用 LoRA，可以改为 from shared.Lora_model import LoraModel

# import fl_pb2
# import fl_pb2_grpc

# # --- 配置参数 ---
# CLIENT_ID = os.environ.get("CLIENT_ID", f"Client-{random.randint(100,999)}")
# SERVER_ADDR = os.environ.get("SERVER_ADDR", "server:50051")
# TOTAL_ROUNDS = int(os.environ.get("TOTAL_ROUNDS", "5"))
# LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "2"))
# LEARNING_RATE = 0.01

# # FedProx 核心超参数: mu (通常取值 0.01, 0.1, 1.0)
# # mu 越大，本地模型被迫越接近全局模型，适合 Non-IID 程度极高的情况
# MU = float(os.environ.get("FEDPROX_MU", "1")) 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"[{CLIENT_ID}] 当前设备: {device} | FedProx MU: {MU}")

# # --- 工具函数 ---

# def load_local_data():
#     client_folder = CLIENT_ID.lower()
#     data_path = f"./dist_data/{client_folder}/local_data.pt"
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"[{CLIENT_ID}] 未找到数据: {data_path}")
#     local_data = torch.load(data_path, weights_only=False)
#     return DataLoader(local_data, batch_size=32, shuffle=True)

# def serialize_weights(state_dict):
#     buffer = io.BytesIO()
#     torch.save(state_dict, buffer)
#     return buffer.getvalue()

# def deserialize_weights(serialized_bytes):
#     buffer = io.BytesIO(serialized_bytes)
#     return torch.load(buffer, map_location=device)

# # --- 核心训练逻辑 (引入 FedProx) ---

# def local_train(model, global_model_state, train_loader):
#     """引入 FedProx 近端项的训练循环"""
#     model.to(device)
#     optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.5)
#     model.train()
    
#     # 将全局模型状态冻结，用于计算近端项惩罚
#     fixed_global_params = {n: p.clone().detach() for n, p in global_model_state.items()}
    
#     print(f"[{CLIENT_ID}] 开始本地训练 (Samples: {len(train_loader.dataset)})...")
    
#     for epoch in range(LOCAL_EPOCHS):
#         epoch_loss = 0.0
#         correct = 0
#         for data, target in train_loader:
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(data)
            
#             # 1. 原始损失 (CrossEntropy/NLL)
#             base_loss = torch.nn.functional.nll_loss(output, target)
            
#             # 2. FedProx 近端项 (Proximal Term)
#             proximal_term = 0.0
#             for name, param in model.named_parameters():
#                 if name in fixed_global_params:
#                     proximal_term += ( (param - fixed_global_params[name])**2 ).sum()
            
#             # 总损失 = 原始损失 + (mu/2) * 近端项
#             loss = base_loss + (MU / 2) * proximal_term
            
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
            
#         acc = 100. * correct / len(train_loader.dataset)
#         print(f"[{CLIENT_ID}] Epoch {epoch+1}: Loss: {epoch_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")
#     model.cpu() # 训练完成后移回 CPU，准备序列化
#     return model.state_dict(), len(train_loader.dataset)

# # --- 主控循环 ---

# def run():
#     train_loader = load_local_data()
#     model = SimpleCNN().to(device)
    
#     with grpc.insecure_channel(SERVER_ADDR) as channel:
#         stub = fl_pb2_grpc.FederatedLearningStub(channel)
        
#         for current_round in range(1, TOTAL_ROUNDS + 1):
#             print(f"\n[{CLIENT_ID}] --------- 第 {current_round} 轮 ---------")
            
#             # 1. 获取全局模型
#             while True:
#                 resp = stub.GetGlobalModel(fl_pb2.GlobalModelRequest(round_number=current_round))
#                 if resp.is_ready:
#                     global_state = deserialize_weights(resp.global_weights)
#                     model.load_state_dict(global_state)
#                     break
#                 time.sleep(1)
                
#             # 2. 本地训练 (传入全局权重用于 FedProx)
#             local_weights, num_samples = local_train(model, global_state, train_loader)
            
#             # 3. 上传参数
#             upload_req = fl_pb2.LocalModel(
#                 client_id=CLIENT_ID,
#                 model_weights=serialize_weights(local_weights),
#                 num_samples=num_samples,
#                 round_number=current_round
#             )
#             response = stub.UploadLocalModel(upload_req)
#             print(f"[{CLIENT_ID}] 服务器响应: {response.message}")

# if __name__ == '__main__':
#     run()