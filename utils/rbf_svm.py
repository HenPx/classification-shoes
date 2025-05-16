import numpy as np

def predict_svm_rbf(X_train, y_train, X_test, alpha, b, gamma):
    y_pred = []
    for x in X_test:
        result = 0
        for i in range(len(X_train)):
            result += alpha[i] * y_train[i] * rbf_kernel(X_train[i], x, gamma)
        result += b
        y_pred.append(np.sign(result))
    return np.array(y_pred)

def rbf_kernel(x1, x2, gamma):
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))