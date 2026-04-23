import torch
import hashlib
import numpy as np

class BlockchainRandomProjector:
    def __init__(self, target_dim=256, blockhash_seed="0x123abc..."):
        """
        :param target_dim: 降维后的目标维度 (如 256)，维度越小上链越便宜，但距离误差会略微增加
        :param blockhash_seed: 智能合约生成的不可预测随机信标 (区块哈希)
        """
        self.target_dim = target_dim
        # 将链上的哈希字符串转化为确定的 64 位整数 Seed
        self.seed = int(hashlib.sha256(blockhash_seed.encode()).hexdigest()[:16], 16) % (2**32 - 1)

    def _flatten_state_dict(self, state_dict):
        """将复杂的 PyTorch 字典展平为一根巨大的 1D 面条 (一维张量)"""
        # 注意：遍历顺序必须固定，字典默认按插入顺序，为绝对安全建议排序 (可选)
        return torch.cat([v.clone().cpu().flatten() for v in state_dict.values()])

    def project(self, state_dict):
        """
        执行随机投影降维 (使用流式分块防止 OOM)
        """
        flat_weights = self._flatten_state_dict(state_dict)
        original_dim = flat_weights.shape[0]

        # 实例化固定种子的生成器，保证全球所有节点用这同一个 Blockhash 生成的投影矩阵 P 是绝对一样的
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed)

        # 🌟 工业级内存优化：分块 (Chunking) 大小设为 100,000 个参数
        chunk_size = 100000 
        projected_vector = torch.zeros(self.target_dim, dtype=torch.float32, device='cpu')

        # print(f"🚀 开始随机投影降维: [原始维度 {original_dim}] -> [链上维度 {self.target_dim}]")

        # 流式切片，每次只生成一小块投影矩阵，算完即销毁，内存占用趋近于 O(1)
        for i in range(0, original_dim, chunk_size):
            end_i = min(i + chunk_size, original_dim)
            current_chunk_size = end_i - i

            # 提取当前这 10 万个参数
            weight_chunk = flat_weights[i:end_i]

            # 生成当前块的局部投影矩阵 P_chunk 形状: (256, 100000)
            # 根据 JL 引理，使用标准正态分布 N(0, 1)
            P_chunk = torch.empty((self.target_dim, current_chunk_size), dtype=torch.float32).normal_(
                mean=0.0, std=1.0, generator=generator
            )

            # 执行矩阵乘法并累加到最终的轻量级向量中
            # 公式: v_light = P_chunk @ weight_chunk
            projected_vector += torch.matmul(P_chunk, weight_chunk)

        # 为了保持距离的标度一致性，数学上通常需要除以 sqrt(target_dim)
        projected_vector = projected_vector / (self.target_dim ** 0.5)

        return projected_vector

# ==========================================
# 🚀 测试与验证：证明距离保留特性 (JL 引理)
# ==========================================
if __name__ == "__main__":
    from model import SimpleCNN # 假设这是你的模型
    import copy

    # 1. 模拟链上广播的随机数 (Blockhash)
    current_blockhash = "0x88e96d4537bea4d9c05d12549907b32561d3bf31f45aae734cdc119f13406cb6"

      
    projector = BlockchainRandomProjector(target_dim=256, blockhash_seed=current_blockhash)

    # 3. 创建三个模型进行测试
    model_A = SimpleCNN()
    
    # 模拟 Model B: 与 A 非常相似的诚实节点 (给 A 加上极微小的正常训练波动)
    model_B = copy.deepcopy(model_A)
    for param in model_B.parameters():
        param.data += torch.randn_like(param.data) * 0.001
        
    # 模拟 Model C: 恶意投毒节点 (参数充满巨大的随机垃圾)
    model_C = SimpleCNN() 
    for param in model_C.parameters():
        param.data += torch.randn_like(param.data) * 5.0

    # 4. 执行极其消耗资源的本地高维真实距离计算 (几十万维度的欧氏距离)
    flat_A = projector._flatten_state_dict(model_A.state_dict())
    flat_B = projector._flatten_state_dict(model_B.state_dict())
    flat_C = projector._flatten_state_dict(model_C.state_dict())
    
    true_dist_AB = torch.norm(flat_A - flat_B).item()
    true_dist_AC = torch.norm(flat_A - flat_C).item()

    # 5. 执行降维，获取准备上链的 256 维轻量向量
    vec_A = projector.project(model_A.state_dict())
    vec_B = projector.project(model_B.state_dict())
    vec_C = projector.project(model_C.state_dict())
    
    proj_dist_AB = torch.norm(vec_A - vec_B).item()
    proj_dist_AC = torch.norm(vec_A - vec_C).item()

    print("\n--- 距离保持特性测试 ---")
    print(f"✅ 诚实节点对 (A vs B):")
    print(f"   真实的百万维距离: {true_dist_AB:.4f}")
    print(f"   降维后 256维距离: {proj_dist_AB:.4f}")
    
    print(f"\n❌ 投毒节点对 (A vs C):")
    print(f"   真实的百万维距离: {true_dist_AC:.4f}")
    print(f"   降维后 256维距离: {proj_dist_AC:.4f}")