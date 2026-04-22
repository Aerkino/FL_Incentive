# 使用轻量级 Python 镜像
FROM python:3.9-slim

WORKDIR /app

# 安装 gRPC 运行时依赖
RUN pip install grpcio grpcio-tools
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 将生成的桩代码和主程序复制进去
COPY client/fl_pb2.py .
COPY client/fl_pb2_grpc.py .
COPY client/main.py .
COPY shared/model.py .
COPY client/local_baseline.py .

# 暴露 gRPC 默认端口
EXPOSE 50051

CMD ["python", "-u", "main.py"]