# 线性回归：最小二乘法 + 梯度下降 两种实现
import numpy as np

class LinearRegression:
    def __init__(self, method='ls'):
        self.method = method  # 'ls' 最小二乘, 'gd' 梯度下降
        self.w = None

    def fit(self, X, y, lr=0.01, n_iter=500):
        # 加偏置列
        X = np.column_stack([np.ones(len(X)), X])

        if self.method == 'ls':
            # 正规方程: w = (X^T X)^(-1) X^T y
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        else:
            # 梯度下降
            self.w = np.zeros(X.shape[1])
            for _ in range(n_iter):
                grad = -2 * X.T @ (y - X @ self.w) / len(y)
                self.w -= lr * grad

    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X])
        return X @ self.w

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


if __name__ == '__main__':
    # 简单测试：y = 3 + 2*x + 噪声
    np.random.seed(42)
    X = np.random.rand(50, 1) * 10
    y = 3.0 + 2.0 * X.flatten() + np.random.randn(50) * 2

    for method in ['ls', 'gd']:
        model = LinearRegression(method=method)
        model.fit(X, y)
        r2 = model.score(X, y)
        print(f'{method}: w = {model.w}, R^2 = {r2:.4f}')
