# import numpy as np

# class SecureAggregationSMC:
#     def __init__(self, num_nodes, model_dim):
#         self.num_nodes = num_nodes
#         self.model_dim = model_dim
#         # 在真实应用中，这些种子是通过 Diffie-Hellman 密钥交换生成的
#         # 这里为了模拟，我们直接用一个字典来存储节点间的共享掩码
#         self.pairwise_masks = {}
#         self._generate_pairwise_masks()

#     def _generate_pairwise_masks(self):
#         """
#         阶段 1：模拟生成节点间的成对掩码（Pairwise Masks）
#         节点 i 和节点 j 共享一个掩码，i 加它，j 就减它。
#         """
#         for i in range(self.num_nodes):
#             for j in range(i + 1, self.num_nodes):
#                 # 随机生成一个与模型维度相同的噪声向量
#                 noise = np.random.uniform(-10.0, 10.0, self.model_dim)
#                 self.pairwise_masks[(i, j)] = noise
#                 self.pairwise_masks[(j, i)] = -noise # 反向抵消

#     def mask_local_model(self, node_id, local_model):
#         """
#         阶段 2：数据拥有者 (DO) 在本地给模型加噪
#         """
#         masked_model = np.copy(local_model)
#         for other_node in range(self.num_nodes):
#             if other_node != node_id:
#                 # 把和所有其他节点协商的噪声都加进去
#                 masked_model += self.pairwise_masks[(node_id, other_node)]
#         return masked_model

#     def server_aggregate(self, masked_models_dict):
#         """
#         阶段 3：轮值委员会（或中心服务器）进行盲聚合
#         传入的字典 key 为 node_id, value 为加噪后的模型
#         """
#         # 取出第一个模型作为累加基底
#         nodes = list(masked_models_dict.keys())
#         global_model = np.copy(masked_models_dict[nodes[0]])
        
#         for i in range(1, len(nodes)):
#             global_model += masked_models_dict[nodes[i]]
            
#         return global_model


# # ==========================================
# # 🚀 模拟测试流程 (可以直接跑)
# # ==========================================
# if __name__ == "__main__":
#     np.random.seed(42)
    
#     NUM_NODES = 5
#     MODEL_DIM = 10 # 假设模型只有 10 个参数
    
#     # 1. 模拟 5 个节点本地训练出真实的原始模型 (明文)
#     true_local_models = {i: np.random.rand(MODEL_DIM) for i in range(NUM_NODES)}
    
#     print("--- 1. 真实的模型聚合结果 (Baseline) ---")
#     baseline_global = np.sum([model for model in true_local_models.values()], axis=0)
#     print(np.round(baseline_global, 4))
    
#     # 2. 初始化 SMC 引擎
#     smc_engine = SecureAggregationSMC(NUM_NODES, MODEL_DIM)
    
#     # 3. 节点在本地加噪 (这就是他们上传到 IPFS 的东西，纯乱码)
#     masked_models = {}
#     for node_id in range(NUM_NODES):
#         masked_models[node_id] = smc_engine.mask_local_model(node_id, true_local_models[node_id])      
#         print(f"\n--- 2. 节点{node_id}上传到 IPFS 的掩码模型 (纯乱码，防止泄露) ---")
#         print(np.round(true_local_models[node_id], 4))
#         print(np.round(masked_models[node_id], 4))
    
#     # 4. 轮值委员会从 IPFS 下载掩码模型并聚合
#     print("\n--- 3. 委员会通过 SMC 盲聚合的结果 ---")
#     smc_global = smc_engine.server_aggregate(masked_models)
#     print(np.round(smc_global, 4))
    
#     # 5. 验证是否完全一致
#     difference = np.linalg.norm(baseline_global - smc_global)
#     print(f"\n✅ 聚合误差 (应趋近于 0): {difference:.10f}")


import hashlib
import random

# ==========================================
# 0. 密码学全局参数 (公开的)
# ==========================================
# 在真实的 Web3 环境中，通常使用椭圆曲线 (如 secp256k1)
# 为了模拟演示的直观性，这里使用传统的素数域 Diffie-Hellman 参数
P = 9973  # 一个公开的大素数 (Modulus)
G = 2     # 公开的生成元 (Base)

# ==========================================
# 1. 链上智能合约 (纯模拟)
# 作用：只负责存储和广播公钥，绝对不碰私钥和真实数据
# ==========================================
class FLSmartContract:
    def __init__(self):
        self.public_keys_registry = {}  # 链上状态：存储节点的公钥

    def register_public_key(self, node_id, public_key):
        """链上交易：节点注册自己的公钥"""
        self.public_keys_registry[node_id] = public_key
        # 在以太坊中，这里会触发一个 Event: LogPublicKeyRegistered
        print(f"[智能合约] 节点 {node_id} 的公钥 {public_key} 已上链记录。")

    def get_all_public_keys(self):
        """链上视图函数：供所有人读取当前的公钥目录"""
        return self.public_keys_registry


# ==========================================
# 2. 客户端 (数据拥有者 DO)
# 作用：本地生成私钥，链上发公钥，链下算种子
# ==========================================
class FLClient:
    def __init__(self, node_id):
        self.node_id = node_id
        # 1. 本地生成极其机密的私钥 (绝对不出本地)
        self.__private_key = random.randint(1, P - 1)
        # 2. 算出公钥 (准备上链) -> G^private_key mod P
        self.public_key = pow(G, self.__private_key, P)
        
        self.shared_seeds = {} # 存储与其他节点协商出的 32 字节种子

    def generate_shared_seeds(self, on_chain_pub_keys):
        """
        核心逻辑：从链上拉取公钥，在本地隔空计算出对等密钥 (SMC Seed)
        """
        for other_node_id, other_pub_key in on_chain_pub_keys.items():
            if other_node_id != self.node_id:
                # Diffie-Hellman 魔法：(Other_PubKey)^My_PrivateKey mod P
                shared_secret = pow(other_pub_key, self.__private_key, P)
                
                # 工业界标准做法：不直接用 DH 秘密，而是做一次哈希派生出 32 字节种子
                # 排序 node_id 保证 A算B 和 B算A 的哈希输入完全一致
                pair_string = f"SMC_SEED_{min(self.node_id, other_node_id)}_{max(self.node_id, other_node_id)}_{shared_secret}"
                seed_32bytes = hashlib.sha256(pair_string.encode()).hexdigest()
                
                self.shared_seeds[other_node_id] = seed_32bytes


# ==========================================
# 🚀 模拟测试流程：5 个参与方的完整生命周期
# ==========================================
if __name__ == "__main__":
    NUM_CLIENTS = 5
    
    # 实例化智能合约 (部署在链上)
    contract = FLSmartContract()
    
    # 实例化 5 个客户端
    clients = [FLClient(node_id=i) for i in range(NUM_CLIENTS)]
    
    print("--- 阶段 1：节点注册与公钥上链 ---")
    for client in clients:
        # 客户端构建交易，把公钥发给智能合约
        contract.register_public_key(client.node_id, client.public_key)
        
    print("\n--- 阶段 2：链下 Diffie-Hellman 种子协商 ---")
    # 客户端监听合约，获取所有人的公钥目录
    on_chain_registry = contract.get_all_public_keys()
    
    # 客户端各自在本地闭门计算种子 (全程不需要互相通信！)
    for client in clients:
        client.generate_shared_seeds(on_chain_registry)
    print("所有节点已在本地计算完共享种子。")

    print("\n--- 阶段 3：零信任验证 (密码学奇迹) ---")
    # 让我们来验证一下 节点 1 和 节点 4 的种子是否一致
    node_1 = clients[1]
    node_4 = clients[4]
    
    seed_1_for_4 = node_1.shared_seeds[4]
    seed_4_for_1 = node_4.shared_seeds[1]
    
    print(f"节点 1 为节点 4 算出的种子: {seed_1_for_4[:15]}...")
    print(f"节点 4 为节点 1 算出的种子: {seed_4_for_1[:15]}...")
    
    if seed_1_for_4 == seed_4_for_1:
        print("\n✅ 验证成功！节点 1 和 节点 4 在未暴露私钥、未直接通信的情况下，拿到了完全相同的 SMC 种子！")
        print("之后他们就可以把这个种子喂给伪随机数生成器(PRG)来生成相同的噪声矩阵进行加减抵消了。")
    else:
        print("❌ 验证失败！")