import os

def generate_compose_file(num_clients, total_rounds=5, local_epochs=2):
    # 基础结构 (Server 部分)
    yaml_content = f"""version: '3.8'

services:
  server:
    build: 
      context: .                  # 【关键】把上帝视角提升到根目录
      dockerfile: server/Dockerfile # 指明 Dockerfile 的具体位置
    ports:
      - "50051:50051"
    environment:
      - MIN_CLIENTS={num_clients} # 服务器需要等待的最少客户端数量
    volumes:
      - ./dist_data/server:/app/data:ro
    networks:
      - fl_net
"""

    # 动态生成 Client 部分
    for i in range(num_clients):
        yaml_content += f"""
  client_{i}:
    build: 
      context: .                  # 同样提升到根目录
      dockerfile: client/Dockerfile # 指明 Dockerfile 的具体位置
    environment:
      - CLIENT_ID=Client_{i}
      - SERVER_ADDR=server:50051
      - TOTAL_ROUNDS={total_rounds}   # 统一下发总轮次
      - LOCAL_EPOCHS={local_epochs}   # 统一下发本地 Epoch
    volumes:
      - ./dist_data/client_{i}:/app/data:ro
      - ./dist_data/server:/app/global_test:ro  
    depends_on:
      - server
    networks:
      - fl_net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all      # 如果你想限制使用的显卡数量，可以把 all 改成 1
              capabilities: [gpu]
"""

    # 结尾网络配置
    yaml_content += """
networks:
  fl_net:
    driver: bridge
"""

    # 写入文件
    with open("docker-compose.yml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print(f"✅ 已成功生成包含 {num_clients} 个客户端的 docker-compose.yml 文件！")

if __name__ == "__main__":
    generate_compose_file(num_clients=5, total_rounds=10, local_epochs=1) # 你可以随意改成 10, 50, 甚至 100