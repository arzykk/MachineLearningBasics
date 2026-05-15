import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-z))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    y_train0 = y_train.copy()
    y_train0[y_train != 0] = 0
    y_train0[y_train == 0] = 1
    lrgd0 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd0.fit(X_train, y_train0)

    y_train1 = y_train.copy()
    y_train1[y_train != 1] = 0
    y_train1[y_train == 1] = 1
    lrgd1 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd1.fit(X_train, y_train1)

    y_train2 = y_train.copy()
    y_train2[y_train != 2] = 0
    y_train2[y_train == 2] = 1
    lrgd2 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd2.fit(X_train, y_train2)

    print(y_train)
    print(y_train0)

    print(lrgd0.activation(lrgd0.net_input(X_test)))
    print(lrgd1.activation(lrgd1.net_input(X_test)))
    print(lrgd2.activation(lrgd2.net_input(X_test)))

    prob0 = lrgd0.activation(lrgd0.net_input(X_test))
    prob1 = lrgd1.activation(lrgd1.net_input(X_test))
    prob2 = lrgd2.activation(lrgd2.net_input(X_test))

    probs = np.vstack((prob0, prob1, prob2)).T

    y_pred = np.argmax(probs, axis=1)

    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)

    print("y_test:", y_test)
    print("y_pred:", y_pred)

    plot_decision_regions(X=X_test, y=y_test, classifier=lrgd2)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()



    # --- siatka punktów ---
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, 0.02),
        np.arange(x2_min, x2_max, 0.02)
    )

    grid = np.array([xx1.ravel(), xx2.ravel()]).T

    # --- predykcja OvR ---
    prob0 = lrgd0.activation(lrgd0.net_input(grid))
    prob1 = lrgd1.activation(lrgd1.net_input(grid))
    prob2 = lrgd2.activation(lrgd2.net_input(grid))

    probs = np.vstack((prob0, prob1, prob2)).T
    Z = np.argmax(probs, axis=1)
    Z = Z.reshape(xx1.shape)

    # --- rysowanie tła (decision regions) ---
    plt.contourf(xx1, xx2, Z, alpha=0.3)

    # --- train set ---
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        marker='o',
        edgecolor='black',
        label='train'
    )

    # --- test set (wyróżniony) ---
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        marker='s',
        edgecolor='black',
        s=100,
        label='test'
    )

    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend()
    plt.title('Logistic Regression OvR - Decision Regions')
    plt.show()




if __name__ == '__main__':
    main()
