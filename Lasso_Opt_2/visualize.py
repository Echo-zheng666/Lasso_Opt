import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- 全局配置 ----------------------
# 绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 算法样式映射（含ADMM不同rho）
STYLE_MAP = {
    # 基础算法
    'GD': ('#1f77b4', '-', 'o', 'GD'),
    'SGD': ('#ff7f0e', '--', 's', 'SGD'),
    'PGD': ('#2ca02c', '-.', '^', 'PGD'),
    'FISTA': ('#d62728', '-', 'x', 'FISTA'),
    'APGD': ('#9467bd', '--', 'd', 'APGD'),
    'CD': ('#8c564b', '-', '*', 'CD'),
    # ADMM不同rho
    'ADMM_0.5': ('#e377c2', '-.', 'p', 'ADMM(rho=0.5)'),
    'ADMM_1.0': ('#7f7f7f', '-.', 'h', 'ADMM(rho=1.0)'),
    'ADMM_5.0': ('#bcbd22', '-.', '8', 'ADMM(rho=5.0)')
}

# ---------------------- 数据加载 ----------------------
def load_loss_data(csv_path='experiment_results_with_loss.csv'):
    """加载并解析损失数据"""
    df = pd.read_csv(csv_path)
    
    # 解析loss_history为浮点数列表（鲁棒处理异常值）
    def parse_loss_hist(hist_str):
        if pd.isna(hist_str) or hist_str.strip() == '':
            return []
        try:
            return [float(x.strip()) for x in hist_str.split(',') if x.strip() != '']
        except Exception as e:
            print(f"解析损失历史时出错：{e}")
            return []
    
    df['loss_history'] = df['loss_history'].apply(parse_loss_hist)
    # 过滤空的损失历史
    df = df[df['loss_history'].apply(len) > 0]
    return df

# ---------------------- 绘制单张收敛图 ----------------------
def plot_convergence_single(df, n_target, p_target, max_iter_show=500):
    """
    绘制指定(n,p)的收敛曲线（包含所有算法+ADMM三种rho）
    :param df: 实验结果DataFrame
    :param n_target: 目标样本数
    :param p_target: 目标特征数
    :param max_iter_show: 展示的最大迭代次数
    """
    # 筛选目标(n,p)的数据
    df_target = df[(df['n'] == n_target) & (df['p'] == p_target)].copy()
    if df_target.empty:
        print(f"未找到(n={n_target}, p={p_target})的实验数据！")
        return
    
    # 绘图
    plt.figure(figsize=(14, 8))
    
    # 绘制基础算法
    for alg in ['GD', 'SGD', 'PGD', 'FISTA', 'APGD', 'CD']:
        df_alg = df_target[df_target['algorithm'] == alg]
        if df_alg.empty:
            continue
        row = df_alg.iloc[0]
        loss_hist = row['loss_history'][:max_iter_show]
        iters = np.arange(1, len(loss_hist) + 1)
        color, linestyle, marker, label = STYLE_MAP[alg]
        
        plt.plot(iters, loss_hist, label=label, color=color, linestyle=linestyle,
                 marker=marker, markersize=4, linewidth=2, alpha=0.8)
    
    # 绘制ADMM不同rho
    df_admm = df_target[df_target['algorithm'] == 'ADMM']
    for _, row in df_admm.iterrows():
        rho = row['rho']
        alg_key = f'ADMM_{rho}'
        if alg_key not in STYLE_MAP:
            continue
        loss_hist = row['loss_history'][:max_iter_show]
        iters = np.arange(1, len(loss_hist) + 1)
        color, linestyle, marker, label = STYLE_MAP[alg_key]
        
        plt.plot(iters, loss_hist, label=label, color=color, linestyle=linestyle,
                 marker=marker, markersize=4, linewidth=2, alpha=0.8)
    
    # 图表属性
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Loss Value (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.title(f'Convergence Speed Comparison (n={n_target}, p={p_target})', fontsize=14)
    plt.legend(loc='upper right', frameon=True, fancybox=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'convergence_n{n_target}_p{p_target}.png', bbox_inches='tight')
    plt.show()

# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    # 加载数据
    df = load_loss_data('experiment_results_with_loss.csv')
    
    # 绘制第一张图：n=500, p=100,低维常规
    plot_convergence_single(df, n_target=500, p_target=100, max_iter_show=1000)
    
    # 绘制第二张图：n=100, p=500，高维稀疏
    plot_convergence_single(df, n_target=100, p_target=500, max_iter_show=500)