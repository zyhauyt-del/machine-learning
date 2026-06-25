# K-Means 聚类 手写实现
import numpy as np


def kmeans(X, k, max_iter=300, tol=1e-4, seed=42):
    """标准K-Means: 随机初始化 → 分配 → 更新中心 → 收敛"""
    rng = np.random.RandomState(seed)
    n, d = X.shape

    # 随机选k个点作为初始中心
    centers = X[rng.choice(n, k, replace=False)].astype(float)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # E步: 分配
        dists = np.zeros((n, k))
        for j in range(k):
            dists[:, j] = np.sum((X - centers[j])**2, axis=1)
        new_labels = np.argmin(dists, axis=1)

        # M步: 更新中心
        new_centers = np.array([X[new_labels == j].mean(axis=0) if np.sum(new_labels == j) > 0
                                else X[rng.choice(n)] for j in range(k)])

        if np.sum((new_centers - centers)**2) < tol:
            break
        centers, labels = new_centers, new_labels

    return labels, centers


def silhouette_score(X, labels):
    """轮廓系数 [-1,1]，越接近1聚类越好"""
    n = len(X)
    unique_lbls = np.unique(labels)
    if len(unique_lbls) < 2 or n < 2:
        return 0.0

    sc_vals = []
    for i in range(n):
        same = labels == labels[i]
        a_i = np.mean([np.sqrt(np.sum((X[i] - X[j])**2)) for j in range(n) if same[j] and j != i]) if same.sum() > 1 else 0

        b_i = float('inf')
        for lbl in unique_lbls:
            if lbl == labels[i]:
                continue
            other = labels == lbl
            b_other = np.mean([np.sqrt(np.sum((X[i] - X[j])**2)) for j in range(n) if other[j]])
            b_i = min(b_i, b_other)

        sc_vals.append((b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0)

    return np.mean(sc_vals)


if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
    labels, centers = kmeans(X, k=3)
    sc = silhouette_score(X, labels)

    print(f'聚类完成: K=3, 轮廓系数={sc:.4f}')
    for i in range(3):
        print(f'  簇{i+1}: {np.sum(labels==i)} 个样本, 中心=({centers[i][0]:.2f}, {centers[i][1]:.2f})')
