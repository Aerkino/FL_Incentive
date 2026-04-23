import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# --- 1. 参数初始化 ---
N = 20
omega = 100.0
np.random.seed(42)
c = np.sort(np.random.uniform(0.5, 5.0, N)) # 节点本地成本

# --- 2. 求解 DO 的隐式纳什均衡总分 S ---
def equation_S(S, alpha, B, c):
    """即您的核心方程: \sum ((alpha*S + B) / (c_i*S^2 + B)) - 1 = 0"""
    return np.sum((alpha * S + B) / (c * S**2 + B)) - 1

def find_equilibrium_S(alpha, B, c):
    if alpha <= 0 and B <= 0: return 1e-5
    try:
        # 动态寻找上界，因为 F(S) 单调递减且跨越 0
        high = 1.0
        while equation_S(high, alpha, B, c) > 0:
            high *= 2
        # Brentq 求根算法，比牛顿法更稳定
        return optimize.brentq(equation_S, 1e-5, high, args=(alpha, B, c))
    except:
        return 1e-5

# --- 3. 任务发布者 (TP) 全局效用函数 ---
def tp_utility(alpha, B, c, omega):
    S = find_equilibrium_S(alpha, B, c)
    return omega * np.log(1 + S) - alpha * S - B

def objective(x, c, omega):
    return -tp_utility(x[0], x[1], c, omega)

# --- 4. 斯塔克尔伯格领导者优化 (寻优最佳 alpha, B) ---
# 给定合理边界防止退化
res = optimize.minimize(objective, [1.0, 1.0], args=(c, omega), bounds=[(0.01, 15), (0.01, 15)])
opt_alpha, opt_B = res.x
opt_S = find_equilibrium_S(opt_alpha, opt_B, c)

print(f"Optimal Alpha: {opt_alpha:.4f}, Optimal B: {opt_B:.4f}")
print(f"Equilibrium S: {opt_S:.4f}, Max TP Utility: {-res.fun:.4f}")

# 个体最优策略验证:
si_stars = opt_S * (opt_alpha * opt_S + opt_B) / (c * opt_S**2 + opt_B)