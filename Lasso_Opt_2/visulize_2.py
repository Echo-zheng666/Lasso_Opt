import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 配置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

STYLE_MAP = {
    'GD': ('#1f77b4', '-', 'o', 'GD'),
    'SGD': ('#ff7f0e', '--', 's', 'SGD'),
    'PGD': ('#2ca02c', '-.', '^', 'PGD'),
    'FISTA': ('#d62728', '-', 'x', 'FISTA'),
    'APGD': ('#9467bd', '--', 'd', 'APGD'),
    'CD': ('#8c564b', '-', '*', 'CD'),
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

# ---------------------- 绘制单张收敛图（优化CD震荡展示） ----------------------
def plot_convergence_single_optimized(df, n_target, p_target, max_iter_show=100):
    """
    优化CD算法震荡展示：改用线性刻度+截断迭代+收敛标记
    :param max_iter_show: 仅展示前N次迭代（聚焦收敛阶段）
    """
    df_target = df[(df['n'] == n_target) & (df['p'] == p_target)].copy()
    if df_target.empty:
        print(f"未找到(n={n_target}, p={p_target})的实验数据！")
        return
    
    plt.figure(figsize=(14, 8))
    
    # 存储CD收敛信息
    cd_converge_iter = None
    cd_converge_loss = None
    
    # 绘制基础算法（仅展示前max_iter_show次迭代）
    for alg in ['GD', 'SGD', 'PGD', 'FISTA', 'APGD', 'CD']:
        df_alg = df_target[df_target['algorithm'] == alg]
        if df_alg.empty:
            continue
        row = df_alg.iloc[0]
        # 仅取前max_iter_show次迭代的损失
        loss_hist = row['loss_history'][:max_iter_show]
        iters = np.arange(1, len(loss_hist) + 1)
        color, linestyle, marker, label = STYLE_MAP[alg]
        
        # （可选）对CD损失做滑动平均，弱化震荡
        if alg == 'CD':
            window_size = 3  # 滑动窗口大小
            loss_hist = np.convolve(loss_hist, np.ones(window_size)/window_size, mode='same')
        
        # 绘制曲线
        plt.plot(iters, loss_hist, label=label, color=color, linestyle=linestyle,
                 marker=marker, markersize=4, linewidth=2, alpha=0.8)
        
        # 标记CD收敛点
        if alg == 'CD':
            for i in range(1, len(loss_hist)):
                if abs(loss_hist[i] - loss_hist[i-1]) < 1e-4:  # 放宽收敛阈值
                    cd_converge_iter = i + 1
                    cd_converge_loss = loss_hist[i]
                    break
            if cd_converge_iter:
                plt.scatter(cd_converge_iter, cd_converge_loss, color=color, s=150,
                            marker='*', edgecolor='black', label='CD Convergence Point', zorder=5)
                plt.axvline(x=cd_converge_iter, color=color, linestyle='--', alpha=0.5)
                plt.text(cd_converge_iter + 2, cd_converge_loss, 
                         f'Converge at iter {cd_converge_iter}', color=color, fontsize=10)
    
    # 绘制ADMM
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
    
    # 图表属性：改用线性刻度
    plt.xlabel('Iteration Number (First 100 Iterations)', fontsize=12)
    plt.ylabel('Loss Value (Linear Scale)', fontsize=12)
    # 固定y轴范围到CD收敛后的损失区间
    if cd_converge_loss:
        plt.ylim(cd_converge_loss * 0.8, max(loss_hist) * 1.2)
    plt.title(f'Convergence Speed (n={n_target}, p={p_target}) - Focus on Early Iterations', fontsize=14)
    plt.legend(loc='best', frameon=True, fancybox=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'convergence_optimized_n{n_target}_p{p_target}.png', bbox_inches='tight')
    plt.show()
    
# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    # 加载数据
    df = load_loss_data('experiment_results_with_loss.csv')
    
    # 绘制第一张图：n=500, p=100,低维常规
    plot_convergence_single_optimized(df, n_target=500, p_target=100, max_iter_show=1000)