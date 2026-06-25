# 决策树 CART 分类器 (Python实现)
import numpy as np
from collections import Counter


class DecisionTree:
    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    def _gini(self, y):
        # Gini不纯度
        counts = Counter(y)
        prob = np.array(list(counts.values())) / len(y)
        return 1 - np.sum(prob ** 2)

    def _best_split(self, X, y):
        best_gain = 0
        best_feat, best_thresh = None, None

        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            if len(thresholds) > 100:  # 太密集就采样
                thresholds = np.percentile(thresholds, np.linspace(0, 100, 50))

            for thresh in thresholds:
                left = y[X[:, feat] <= thresh]
                right = y[X[:, feat] > thresh]
                if len(left) < self.min_samples or len(right) < self.min_samples:
                    continue

                gain = self._gini(y) - (
                    len(left)/len(y) * self._gini(left) +
                    len(right)/len(y) * self._gini(right)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def _build(self, X, y, depth):
        # 递归建树
        n_samples = len(y)
        if depth >= self.max_depth or n_samples < 2 * self.min_samples or len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain == 0:
            return Counter(y).most_common(1)[0][0]

        left_idx = X[:, feat] <= thresh
        right_idx = ~left_idx
        return {
            'feat': feat, 'thresh': thresh,
            'left': self._build(X[left_idx], y[left_idx], depth + 1),
            'right': self._build(X[right_idx], y[right_idx], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feat']] <= node['thresh']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tree = DecisionTree(max_depth=4)
    tree.fit(X_train, y_train)
    acc = np.mean(tree.predict(X_test) == y_test)
    print(f'CART决策树 — Iris 测试准确率: {acc:.2%}')
