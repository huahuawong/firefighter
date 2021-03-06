# the imports

import pandas as pd
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers


def gaussian_kernel(x, y, sigma=6.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


class SVM(object):

    def __init__(self, kernel=gaussian_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == gaussian_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))


if __name__ == "__main__":
    import pylab as pl


    def gen_non_lin_separable_data():
        # load the data
        wildfiredata0 = pd.read_csv(..csv file...).to_dict()

        wildfiredata1 = pd.read_csv(..csv file...).to_dict()

        # get keys
        x1a = wildfiredata0.get('Daily Hi Temp')
        x2a = wildfiredata1.get('Daily Hi Temp')

        x1a.values()
        x2a.values()

        # normalize the data
        X_1a_untx = np.array(list(x1a.values()))
        X_1a_untx = X_1a_untx / np.max(X_1a_untx)

        X_2a_untx = np.array(list(x2a.values()))
        X_2a_untx = X_2a_untx / np.max(X_2a_untx)

        #Reshaping data into arrays
        X1a_txp = X_1a_untx.reshape(270, 1)
        X2a_txp = X_2a_untx.reshape(270, 1)

        X1 = np.hstack((X1a_txp, np.atleast_2d(X1b_txp), np.atleast_2d(X1c_txp)))
        X2 = np.hstack((X2a_txp, np.atleast_2d(X2b_txp), np.atleast_2d(X2c_txp)))

        y1 = np.ones(len(x1a.values()))* -1
        y2 = np.ones(len(x2a.values()))
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[120:]
        y1_train = y1[120:]
        X2_train = X2[120:]
        y2_train = y2[120:]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train


    def split_test(X1, y1, X2, y2):
        X1_test = X1[:81]
        y1_test = y1[:81]
        X2_test = X2[:81]
        y2_test = y2[:81]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test


    def plot_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")
        pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

        # w.x + b = 0
        a0 = -4;
        a1 = f(a0, clf.w, clf.b)
        b0 = 4;
        b1 = f(b0, clf.w, clf.b)
        pl.plot([a0, b0], [a1, b1], "k")

        # w.x + b = 1
        a0 = -4;
        a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4;
        b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0, b0], [a1, b1], "k--")

        # w.x + b = -1
        a0 = -4;
        a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4;
        b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0, b0], [a1, b1], "k--")

        pl.axis("tight")
        pl.show()


    def plot_contour(X1_train, X2_train, clf):
        pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")
        pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()


    def run_SVM():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(gaussian_kernel)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf)
        print("F1 score:", f1_score(y_test, y_predict, average='macro'))
        
        
    run_SVM()

