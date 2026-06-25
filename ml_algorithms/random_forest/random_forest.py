# 随机森林 (Bagging + 决策树)
import numpy as np
from collections import Counter


class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def _bootstrap(self, X, y):
        # 有放回采样
        n = len(X)
        idx = np.random.choice(n, n, replace=True)
        return X[idx], y[idx]

    def _build_tree(self, X, y, depth):
        """单棵决策树（随机选特征子集）"""
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < 4 or len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        # 随机选特征子集
        if self.max_features == 'sqrt':
            n_feat_sub = max(1, int(np.sqrt(n_features)))
        else:
            n_feat_sub = n_features

        feat_subset = np.random.choice(n_features, n_feat_sub, replace=False)

        # 遍历子集找最优分割
        best_gain, best_feat, best_thresh = 0, None, None
        for feat in feat_subset:
            values = X[:, feat]
            if len(np.unique(values)) < 2:
                continue
            thresh = np.median(values)
            left = y[values <= thresh]
            right = y[values > thresh]
            if len(left) < 2 or len(right) < 2:
                continue
            # Gini gain
            def gini(y_arr):
                cnt = Counter(y_arr)
                p = np.array(list(cnt.values())) / len(y_arr)
                return 1 - np.sum(p**2)
            gain = gini(y) - (len(left)/len(y)*gini(left) + len(right)/len(y)*gini(right))
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh

        if best_feat is None:
            return Counter(y).most_common(1)[0][0]

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx
        return {
            'feat': best_feat, 'thresh': best_thresh,
            'left': self._build_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self._build_tree(X[right_idx], y[right_idx], depth + 1),
        }

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            Xs, ys = self._bootstrap(X, y)
            self.trees.append(self._build_tree(Xs, ys, 0))

    def _predict_tree(self, x, node):
        if not isinstance(node, dict):
            return node
        return self._predict_tree(x, node['left']) if x[node['feat']] <= node['thresh'] else self._predict_tree(x, node['right'])

    def predict(self, X):
        # 投票
        preds = np.array([[self._predict_tree(x, t) for t in self.trees] for x in X])
        return np.array([Counter(row).most_common(1)[0][0] for row in preds])


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForest(n_trees=20, max_depth=5)
    rf.fit(X_train, y_train)
    acc = np.mean(rf.predict(X_test) == y_test)
    print(f'随机森林 — Iris 测试准确率: {acc:.2%}')
