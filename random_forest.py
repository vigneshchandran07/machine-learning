import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    np.random.seed(12345)
    iris = load_iris()

    X = iris.data
    y = iris.target

    model = RandomForestClassifier()

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    k_fold = 10
    slice_size = X.shape[0] / float(k_fold)

    score_sum = 0
    for i in range(0, k_fold):
        start = int(i * slice_size)
        end = int((i + 1) * slice_size)

        X_train = X[np.append(idx[0:start], idx[end:])]
        y_train = y[np.append(idx[0:start], idx[end:])]

        X_test = X[idx[start:end]]
        y_test = y[idx[start:end]]

        clf = model.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
        score_sum += score

    print("Average Cross Validation Score", score_sum / k_fold)
