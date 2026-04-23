import torch
import hashlib
import numpy as np

class PyTorchSMCEngine:
    def __init__(self, client_id, all_client_ids):
        # 🌟 修复 1：强制全部转小写并去空格，彻底杜绝 ASCII 比较错乱！
        self.client_id = str(client_id).strip().lower()
        self.all_client_ids = sorted([str(x).strip().lower() for x in all_client_ids])
        self.quantize_factor = 1e6 
        
    def _get_mock_pairwise_seed(self, id1, id2):
        pair_str = f"SMC_SEED_{min(id1, id2)}_{max(id1, id2)}_Secret"
        seed_hex = hashlib.sha256(pair_str.encode()).hexdigest()
        return int(seed_hex[:16], 16) % (2**32 - 1)

    def mask_state_dict(self, local_state_dict):
        print(f"[{self.client_id}] (SMC) 正在进行高精度量化加噪...")
        ordered_keys = sorted(local_state_dict.keys())
        masked_dict = {}
        
        for k in ordered_keys:
            # 转为 Int64
            masked_dict[k] = (local_state_dict[k].clone().cpu().double() * self.quantize_factor).to(torch.int64)

        for other_id in self.all_client_ids:
            if other_id == self.client_id:
                continue
                
            shared_seed = self._get_mock_pairwise_seed(self.client_id, other_id)
            prg = np.random.Generator(np.random.PCG64(shared_seed))
            
            for k in ordered_keys:
                shape = masked_dict[k].shape
                # 生成纯 Int64 噪声
                noise_np = prg.integers(low=-1000000, high=1000000, size=shape, dtype=np.int64)
                noise_tensor = torch.tensor(noise_np, dtype=torch.int64, device='cpu')
                
                # 因为前面强制了小写，这里的比较绝对可靠
                if self.client_id > other_id:
                    masked_dict[k] += noise_tensor
                else:
                    masked_dict[k] -= noise_tensor
                    
        print(f"[{self.client_id}] (SMC) 量化加密完成！")
        
        # 🌟 修复 2：绝对不要转回 Float32！直接把纯正的 Int64 字典返回，
        # torch.save (序列化) 完全可以无损保存 Int64 发给 gRPC！
        return masked_dict