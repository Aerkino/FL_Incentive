# 使用轻量级 Python 镜像
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安装 gRPC 运行时依赖
RUN pip install grpcio grpcio-tools

# 将生成的桩代码和主程序复制进去
COPY server/fl_pb2.py .
COPY server/fl_pb2_grpc.py .
COPY server/main.py .
COPY shared/model.py .


# 暴露 gRPC 默认端口
EXPOSE 50051

CMD ["python", "-u", "main.py"]