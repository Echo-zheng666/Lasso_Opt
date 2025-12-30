import numpy as np
import pandas as pd
from lasso_algorithms import (
    gd_lasso, sgd_lasso, pgd_lasso, fista_lasso,
    apgd_lasso, cd_lasso, admm_lasso
)

# 实验配置
SEED = 42
np.random.seed(SEED)
LAM = 0.1
N_P_COMBINATIONS = [
    (50, 50), (50, 100), (50, 500),
    (100, 50), (100, 100), (100, 500),
    (500, 50), (500, 100), (500, 500),
    (1000, 50), (1000, 100), (1000, 500)
]
ADMM_RHO_VALUES = [0.5, 1.0, 5.0]
MAX_ITER = 5000
TOL = 1e-4
LR = 0.01

# 生成模拟数据
def generate_data(n, p, sparsity=0.01):
    w_true = np.zeros(p)
    non_zero_indices = np.random.choice(p, int(sparsity * p), replace=False)
    w_true[non_zero_indices] = np.random.randn(int(sparsity * p))
    X = np.random.randn(n, p)
    eps = np.random.randn(n) * 0.1
    y = X @ w_true + eps
    return X, y, w_true

# 执行单个算法实验
def run_algorithm(alg_name, X, y, lam, **kwargs):
    alg_map = {
        'GD': gd_lasso,
        'SGD': sgd_lasso,
        'PGD': pgd_lasso,
        'FISTA': fista_lasso,
        'APGD': apgd_lasso,
        'CD': cd_lasso,
        'ADMM': admm_lasso
    }
    if alg_name not in alg_map:
        raise ValueError(f"未知算法：{alg_name}")
    return alg_map[alg_name](X, y, lam, **kwargs)

# 主实验流程
def main_experiment():
    results = []
    
    for n, p in N_P_COMBINATIONS:
        print(f"开始实验：n={n}, p={p}")
        X, y, w_true = generate_data(n, p)
        
        # 基础算法对比
        for alg in ['GD', 'SGD', 'PGD', 'FISTA', 'APGD', 'CD']:
            try:
                if alg == 'SGD':
                    # 动态调整batch_size
                    if n < 100:
                        batch_size = min(16, n)
                    elif n < 1000:
                        batch_size = 32
                    else:
                        batch_size = 64
                    
                    w, loss_hist, time_cost = run_algorithm(
                        alg, X, y, LAM,
                        max_iter=MAX_ITER,
                        init_lr=0.01,
                        batch_size=batch_size,
                        tol=TOL,
                        lr_decay=0.995
                    )
                elif alg in ['APGD', 'CD']:
                    w, loss_hist, time_cost = run_algorithm(
                        alg, X, y, LAM,
                        max_iter=MAX_ITER,
                        tol=TOL
                    )
                else:
                    w, loss_hist, time_cost = run_algorithm(
                        alg, X, y, LAM,
                        max_iter=MAX_ITER,
                        lr=LR,
                        tol=TOL
                    )
                
                # 将loss_history转为字符串存储（CSV支持）
                loss_hist_str = ','.join(map(str, loss_hist))
                results.append({
                    'n': n,
                    'p': p,
                    'algorithm': alg,
                    'rho': None,
                    'time_cost': time_cost,
                    'final_loss': loss_hist[-1],
                    'iterations': len(loss_hist),
                    'loss_history': loss_hist_str  # 新增：保存损失序列
                })
            except Exception as e:
                print(f"算法{alg}在n={n},p={p}时出错：{e}")
        
        # ADMM不同rho对比
        for rho in ADMM_RHO_VALUES:
            try:
                w, loss_hist, time_cost = run_algorithm(
                    'ADMM', X, y, LAM, rho=rho, max_iter=MAX_ITER, tol=TOL
                )
                loss_hist_str = ','.join(map(str, loss_hist))
                results.append({
                    'n': n,
                    'p': p,
                    'algorithm': 'ADMM',
                    'rho': rho,
                    'time_cost': time_cost,
                    'final_loss': loss_hist[-1],
                    'iterations': len(loss_hist),
                    'loss_history': loss_hist_str  # 新增：保存损失序列
                })
            except Exception as e:
                print(f"ADMM(rho={rho})在n={n},p={p}时出错：{e}")
    
    # 保存结果到CSV
    df = pd.DataFrame(results)
    df.to_csv('experiment_results_with_loss.csv', index=False)
    print("实验完成，结果已保存到experiment_results_with_loss.csv")

if __name__ == "__main__":
    main_experiment()