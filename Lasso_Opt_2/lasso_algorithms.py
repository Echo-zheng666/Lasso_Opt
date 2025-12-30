import numpy as np
from scipy.optimize import line_search
import time

# 定义Lasso损失函数：1/(2n)||Xw - y||² + λ||w||₁
def lasso_loss(w, X, y, lam):
    n = X.shape[0]
    return 0.5 * np.linalg.norm(X @ w - y, 2)**2 / n + lam * np.linalg.norm(w, 1)

# 梯度计算（仅光滑部分）
def grad(w, X, y):
    n = X.shape[0]
    return X.T @ (X @ w - y) / n

# 软阈值算子
def soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)

# 1. 梯度下降（GD）
def gd_lasso(X, y, lam, max_iter=1000, lr=0.01, tol=1e-6):
    n, p = X.shape
    w = np.zeros(p)
    loss_history = []
    start_time = time.time()
    
    for i in range(max_iter):
        loss = lasso_loss(w, X, y, lam)
        loss_history.append(loss)
        g = grad(w, X, y)
        w = w - lr * g  # 无正则项的梯度下降，实际效果差，仅作对比
        
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            break
    
    time_cost = time.time() - start_time
    return w, loss_history, time_cost

# 1. SGD（改进版，包含init_lr、lr_decay等参数）
def sgd_lasso(X, y, lam, max_iter=1000, init_lr=0.01, batch_size=32, tol=1e-6, lr_decay=0.995):
    n, p = X.shape
    w = np.zeros(p)
    loss_history = []
    start_time = time.time()
    
    for i in range(max_iter):
        # 学习率衰减
        lr = init_lr * (lr_decay ** i)
        # 随机采样（允许重复）
        idx = np.random.choice(n, batch_size, replace=True)
        X_batch = X[idx]
        y_batch = y[idx]
        
        loss = lasso_loss(w, X, y, lam)
        loss_history.append(loss)
        # 计算批次梯度
        g = X_batch.T @ (X_batch @ w - y_batch) / batch_size
        # 近端梯度更新（软阈值）
        w = soft_threshold(w - lr * g, lr * lam)
        
        # 收敛判断（滑动平均）
        if i > 10:
            recent_loss = np.mean(loss_history[-10:])
            prev_recent_loss = np.mean(loss_history[-20:-10])
            if abs(recent_loss - prev_recent_loss) < tol:
                break
    
    time_cost = time.time() - start_time
    return w, loss_history, time_cost

# 3. 近端梯度下降（PGD）
def pgd_lasso(X, y, lam, max_iter=1000, lr=0.01, tol=1e-6):
    n, p = X.shape
    w = np.zeros(p)
    loss_history = []
    start_time = time.time()
    
    for i in range(max_iter):
        loss = lasso_loss(w, X, y, lam)
        loss_history.append(loss)
        # 近端梯度步骤：w = soft_threshold(w - lr * ∇f(w), lr*lam)
        w = soft_threshold(w - lr * grad(w, X, y), lr * lam)
        
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            break
    
    time_cost = time.time() - start_time
    return w, loss_history, time_cost

# 4. 加速近端梯度下降（FISTA）
def fista_lasso(X, y, lam, max_iter=1000, lr=0.01, tol=1e-6):
    n, p = X.shape
    w = np.zeros(p)
    v = w.copy()
    t = 1.0
    loss_history = []
    start_time = time.time()
    
    for i in range(max_iter):
        loss = lasso_loss(w, X, y, lam)
        loss_history.append(loss)
        
        # FISTA迭代步骤
        w_prev = w
        w = soft_threshold(v - lr * grad(v, X, y), lr * lam)
        t_prev = t
        t = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
        v = w + ((t_prev - 1) / t) * (w - w_prev)
        
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            break
    
    time_cost = time.time() - start_time
    return w, loss_history, time_cost

# 5. 自适应近端梯度下降（APGD）
# 2. APGD（无lr参数，自适应学习率）
def apgd_lasso(X, y, lam, max_iter=1000, tol=1e-6):
    n, p = X.shape
    w = np.zeros(p)
    v = w.copy()
    t = 1.0
    loss_history = []
    start_time = time.time()
    
    for i in range(max_iter):
        loss = lasso_loss(w, X, y, lam)
        loss_history.append(loss)
        
        # 线搜索自适应学习率
        g = grad(v, X, y)
        def f(alpha):
            return lasso_loss(soft_threshold(v - alpha * g, alpha * lam), X, y, lam)
        # 优化线搜索：限制alpha的范围（避免过小/过大）
        alpha, _, _, _, _, _ = line_search(f, lambda x: 0, xk=1.0, pk=-1.0)
        lr = alpha if alpha is not None else 0.01  # 若线搜索失败，用默认lr
        lr = np.clip(lr, 1e-5, 0.1)  # 限制学习率范围
        
        # APGD迭代
        w_prev = w
        w = soft_threshold(v - lr * g, lr * lam)
        t_prev = t
        t = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
        v = w + ((t_prev - 1) / t) * (w - w_prev)
        
        # 双重收敛判断：损失变化 + 参数变化
        if i > 10:
          # 滑动平均损失（减少震荡影响）
          recent_loss = np.mean(loss_history[-10:])
          prev_recent_loss = np.mean(loss_history[-20:-10])
          loss_change = abs(recent_loss - prev_recent_loss)
          # 参数变化量
          w_change = np.linalg.norm(w - w_prev)
          # 满足任一条件则停止
          if loss_change < tol or w_change < tol:
              break
    
    time_cost = time.time() - start_time
    return w, loss_history, time_cost

# 6. 坐标下降（CD）
# CD（无lr参数）
def cd_lasso(X, y, lam, max_iter=1000, tol=1e-6):
    n, p = X.shape
    w = np.zeros(p)
    X_T = X.T
    X_norm = np.sum(X**2, axis=0)  # 预计算每个特征的L2范数
    residual = y.copy()  # 残差初始化为y
    loss_history = []
    start_time = time.time()
    
    # 初始化：计算第一轮迭代前的损失值（保证loss_history的完整性）
    initial_loss = lasso_loss(w, X, y, lam)
    loss_history.append(initial_loss)
    
    for i in range(max_iter):
        # 循环更新每个坐标
        for j in range(p):
            if X_norm[j] == 0:
                w_j = 0
            else:
                # 闭式解更新第j个参数
                w_j = soft_threshold(np.dot(X_T[j], residual) / X_norm[j], lam / X_norm[j] * n)
            # 更新残差和参数
            residual += (w[j] - w_j) * X_T[j]
            w[j] = w_j
        
        # 计算当前迭代后的损失值
        current_loss = lasso_loss(w, X, y, lam)
        loss_history.append(current_loss)
        
        # 收敛判断：相邻两次损失值的绝对变化小于阈值，则提前终止
        if abs(loss_history[-1] - loss_history[-2]) < tol:
            print(f"CD算法在第{i+1}次迭代收敛（损失值变化小于{tol}）")
            break
    
    time_cost = time.time() - start_time
    return w, loss_history, time_cost

# 7. ADMM（交替方向乘子法）
def admm_lasso(X, y, lam, rho=1.0, alpha=1.0, max_iter=1000, tol=1e-6):
    n, p = X.shape
    w = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    loss_history = []
    start_time = time.time()
    
    # 预计算：(X^TX + rho*I)^{-1}X^Ty
    X_Ty = X.T @ y
    X_TX = X.T @ X
    inv_mat = np.linalg.inv(X_TX + rho * np.eye(p))
    
    # 自适应阈值（根据特征数放宽）
    r_tol = np.sqrt(p) * tol
    s_tol = np.sqrt(p) * tol

    
    for i in range(max_iter):
        loss = lasso_loss(w, X, y, lam)
        loss_history.append(loss)
        
        # ADMM三步迭代
        # 1. w-update
        w = inv_mat @ (X_Ty + rho * (z - u))
        # 2. z-update
        z_prev = z  # 保存上一轮z用于残差计算
        z = soft_threshold(w + u, lam / rho * alpha)
        # 3. u-update
        u = u + w - z
        
        # 计算归一化的残差
        r = np.linalg.norm(w - z, 2)  # 原始残差
        s = np.linalg.norm(-rho * (z - z_prev), 2)  # 对偶残差
        
        # 收敛判断：使用自适应阈值
        if r < r_tol and s < s_tol:
            break
    
    time_cost = time.time() - start_time
    return w, loss_history, time_cost