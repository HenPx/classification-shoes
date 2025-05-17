import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ManualSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, gamma=0.1, lr=0.001, epochs=1000):
        self.C = C
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs
        self.classifiers = {}

    def rbf_kernel(self, X1, X2):
        diff = X1 - X2
        return np.exp(-self.gamma * np.dot(diff, diff))

    def _train_binary(self, X, y_binary):
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples)
        bias = 0

        # Precompute kernel
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.rbf_kernel(X[i], X[j])

        # Gradient descent
        for epoch in range(self.epochs):
            for i in range(n_samples):
                condition = y_binary[i] * (np.sum(alpha * y_binary * K[:, i]) + bias)
                if condition < 1:
                    alpha[i] += self.lr * (1 - condition)
                else:
                    alpha[i] -= self.lr * self.C * alpha[i]
            alpha = np.clip(alpha, 0, self.C)

        # Support vectors
        support_indices = np.where(alpha > 1e-5)[0]
        support_vectors = X[support_indices]
        support_alpha = alpha[support_indices]
        support_y = y_binary[support_indices]

        # Bias estimate
        bias = 0
        for i in support_indices:
            bias += y_binary[i] - np.sum(alpha * y_binary * K[:, i])
        bias /= len(support_indices)

        return {
            'alpha': support_alpha,
            'bias': bias,
            'support_vectors': support_vectors,
            'support_y': support_y
        }

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers = {}

        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, -1)
            self.classifiers[cls] = self._train_binary(X, y_binary)

        return self

    def _project(self, x, clf):
        result = 0
        for alpha, sv_y, sv in zip(clf['alpha'], clf['support_y'], clf['support_vectors']):
            result += alpha * sv_y * self.rbf_kernel(x, sv)
        return result + clf['bias']

    def predict(self, X):
        scores = []
        for cls in self.classes_:
            clf = self.classifiers[cls]
            class_scores = np.array([self._project(x, clf) for x in X])
            scores.append(class_scores)

        scores = np.stack(scores, axis=1)  # shape: (n_samples, n_classes)
        return self.classes_[np.argmax(scores, axis=1)]