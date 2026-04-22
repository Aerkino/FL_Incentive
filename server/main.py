import grpc
from concurrent import futures
import threading
import os
import logging
import io

# 引入 PyTorch 生态
import torch
from torch.utils.data import DataLoader

# 导入你自己的模型
from model import SimpleCNN

# 导入编译生成的代码 (注意服务名从 Sum 变成了 Learning)
import fl_pb2
import fl_pb2_grpc

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 序列化工具函数 ---
def serialize_weights(state_dict):
    """将 PyTorch state_dict 转化为字节流"""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()

def deserialize_weights(serialized_bytes):
    """将字节流还原为 PyTorch state_dict"""
    buffer = io.BytesIO(serialized_bytes)
    return torch.load(buffer)


class FederatedLearningServicer(fl_pb2_grpc.FederatedLearningServicer):
    def __init__(self):
        self.current_round = 1
        self.min_clients_required = int(os.environ.get("MIN_CLIENTS", "3"))
        self.received_data = {}  
        self.lock = threading.Lock()
        self.is_aggregating = False 

        # ================= 联邦学习专属初始化 =================
        logging.info("初始化全局模型...")
        self.global_model = SimpleCNN()
        
        # 尝试加载全局测试集（由 docker-compose 的 volume 挂载）
        test_data_path = "/app/data/test_data.pt"
        if os.path.exists(test_data_path):
            self.test_dataset = torch.load(test_data_path, weights_only=False)
            self.test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)
            logging.info(f"成功加载全局测试集，样本数: {len(self.test_dataset)}")
        else:
            logging.warning(f"⚠️ 未找到测试集 {test_data_path}，聚合后将跳过评估阶段。")
            self.test_loader = None
        # ======================================================

    def UploadLocalModel(self, request, context):
        trigger_aggregate = False
        
        # --- 极小化临界区 ---
        with self.lock:
            # 1. 拦截过期或超前的请求
            if request.round_number != self.current_round:
                return fl_pb2.ServerResponse(
                    success=False, 
                    message=f"轮次不匹配。服务器当前轮次: {self.current_round}"
                )
                
            # 2. 防抖/去重
            if request.client_id in self.received_data:
                return fl_pb2.ServerResponse(
                    success=False, 
                    message="本轮已收到过您的数据，请勿重复提交"
                )

            # 3. 写入数据 (保存字节流和客户端样本数)
            self.received_data[request.client_id] = {
                'weights_bytes': request.model_weights,
                'num_samples': request.num_samples
            }
            
            # 4. 触发条件检查
            if len(self.received_data) == self.min_clients_required and not self.is_aggregating:
                trigger_aggregate = True
                self.is_aggregating = True
        # --------------------------------------

        logging.info(f"[Round {request.round_number}] 收到来自 {request.client_id} 的模型更新 (用于训练的样本数: {request.num_samples})")

        # 锁释放后，异步触发聚合
        if trigger_aggregate:
            threading.Thread(target=self._aggregate).start()

        return fl_pb2.ServerResponse(success=True, message="模型参数已接收")

    def GetGlobalModel(self, request, context):
        # 读操作临界区
        with self.lock:
            current_round_snapshot = self.current_round
            # 为了防止在极其微小的概率下，读取时正好在发生聚合更新，这里提取一个状态字典的深拷贝快照
            state_dict_snapshot = {k: v.clone() for k, v in self.global_model.state_dict().items()}

        # 核心逻辑：只要客户端请求的轮次 <= 服务器当前轮次，就允许下发全局模型
        if request.round_number <= current_round_snapshot:
            # 在锁外进行耗时的序列化操作，最大化并发性能
            global_weights_bytes = serialize_weights(state_dict_snapshot)
            return fl_pb2.GlobalModelResponse(
                global_weights=global_weights_bytes, 
                is_ready=True
            )
        else:
            return fl_pb2.GlobalModelResponse(is_ready=False)

    def _aggregate(self):
        """执行 FedAvg 联邦平均聚合逻辑"""
        # --- 步骤 1：加锁获取数据快照并清理现场 ---
        with self.lock:
            data_to_compute = self.received_data.copy()
            current_round_processing = self.current_round
            self.received_data = {}
        # ----------------------------------------

        logging.info(f"--- 正在对轮次 {current_round_processing} 的 {len(data_to_compute)} 个节点模型执行 FedAvg ---")
        
        # --- 步骤 2：无锁状态下执行重度计算 (FedAvg) ---
        total_samples = sum([data['num_samples'] for data in data_to_compute.values()])
        global_state_dict = self.global_model.state_dict()
        
        # 将全局模型的临时字典清零，准备累加
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])
            
        # 遍历所有客户端，按数据量加权累加参数
        for client_id, data in data_to_compute.items():
            weight = data['num_samples'] / total_samples
            client_weights = deserialize_weights(data['weights_bytes'])
            
            for key in global_state_dict.keys():
                global_state_dict[key] += client_weights[key] * weight

        # 将聚合后的参数加载回真正的全局模型
        self.global_model.load_state_dict(global_state_dict)
        
        # 聚合完成后，立即进行全局测试集评估
        if self.test_loader is not None:
            self._evaluate()
        # ----------------------------------------
        
        # --- 步骤 3：加锁更新全局状态 ---
        with self.lock:
            self.current_round += 1
            self.is_aggregating = False
        # ----------------------------------------
        
        logging.info(f"--- 轮次 {current_round_processing} 聚合彻底完成！已开启第 {self.current_round} 轮 ---")

    def _evaluate(self):
        """在全局测试集上评估当前模型精度"""
        self.global_model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.global_model(data)
                test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        logging.info(f"⭐ [全局评估] 测试集 Loss: {test_loss:.4f} | 准确率: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%) ⭐")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # 注意这里注册的服务名称改成了 FederatedLearningServicer
    fl_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedLearningServicer(), server)
    server.add_insecure_port('[::]:50051')
    logging.info("联邦学习大脑 (Server) 已启动，监听端口 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()