# KNN 分类器 手写实现
import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        # KNN不需要训练，直接存数据
        self.X = X
        self.y = y

    def _distance(self, a, b):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((a - b)**2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(a - b))
        else:
            return np.sqrt(np.sum((a - b)**2))

    def _predict_one(self, x):
        # 算到所有训练样本的距离，取k近邻投票
        dists = np.array([self._distance(x, xi) for xi in self.X])
        knn_idx = np.argsort(dists)[:self.k]
        knn_labels = self.y[knn_idx]
        return Counter(knn_labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for k in [1, 3, 5, 7]:
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        acc = np.mean(knn.predict(X_test) == y_test)
        print(f'KNN (k={k}) — Iris 测试准确率: {acc:.2%}')
