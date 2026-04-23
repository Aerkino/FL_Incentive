#!/bin/bash

# 当你按下 Ctrl+C 时，自动清理所有后台挂起的 Python 进程
trap 'kill $(jobs -p); echo "所有节点已关闭"; exit' SIGINT SIGTERM

echo "🚀 准备启动联邦学习集群..."

# 1. 启动服务器 (放到后台执行 '&')
export MIN_CLIENTS=5
python server/main.py &
SERVER_PID=$!
echo "✅ Server 已启动 (PID: $SERVER_PID)"

# 稍微等 2 秒，确保服务器的端口 50051 已经监听
sleep 2 

# 2. 循环启动 3 个客户端
for i in {0..4}
do
    export CLIENT_ID="Client_$i"
    export SERVER_ADDR="localhost:50051"
    export TOTAL_ROUNDS="100"
    python client/main.py &
    echo "✅ 客户端 $CLIENT_ID 已启动"
done

echo "🔥 集群运行中！请观察上方输出的日志..."

# 等待所有后台进程结束
wait