# 逻辑回归：Sigmoid + 交叉熵 + 梯度下降
import numpy as np

def sigmoid(z):
    z = np.clip(z, -50, 50)  # 防溢出
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None

    def fit(self, X, y):
        X = np.column_stack([np.ones(len(X)), X])  # 加偏置
        self.w = np.zeros(X.shape[1])

        for _ in range(self.n_iter):
            y_hat = sigmoid(X @ self.w)
            grad = X.T @ (y_hat - y) / len(y)
            self.w -= self.lr * grad

    def predict_proba(self, X):
        X = np.column_stack([np.ones(len(X)), X])
        return sigmoid(X @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(lr=0.1, n_iter=2000)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f'测试准确率: {acc:.2%}')
    print(f'权重: {model.w}')
