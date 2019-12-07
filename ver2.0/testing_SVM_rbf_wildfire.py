# the imports
import pandas as pd
import numpy as np
import numpy.linalg as la
import cvxopt
import cvxopt.solvers
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix


def gaussian_kernel(gamma=7.0):
    return lambda x, y: np.exp(-gamma * la.norm(np.subtract(x, y)))


class SVM(object):

    def __init__(self, kernel=gaussian_kernel, gamma=7, C=1):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)
        self.gamma = gamma
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        # Set gamma to 1/n_features by default
        if not self.gamma:
            self.gamma = 1 / n_features

        # Initialize kernel method with parameters
        self.kernel = self.kernel(
            gamma=self.gamma)

        # Calculate kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        # Define the quadratic optimization problem
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = lagr_mult > 1e-7
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        # Get the samples that will act as support vectors
        self.support_vectors = X[idx]
        # Get the corresponding labels
        self.support_vector_labels = y[idx]

        # Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)


if __name__ == "__main__":
    import pylab as pl

    def gen_non_lin_separable_data():
        # load the data
        wildfiredata0 = pd.read_csv("C:\\Users\\User\\Desktop\\IUPUI Masters\\Fall 2019\\Art Intel\\AI project"
                                    "\\Wildfires_0_Class.csv").to_dict()

        wildfiredata1 = pd.read_csv("C:\\Users\\User\\Desktop\\IUPUI Masters\\Fall 2019\\Art Intel\\AI project"
                                    "\\Wildfires_1_Class.csv").to_dict()

        x1a = wildfiredata0.get('MaxTemp')
        x1b = wildfiredata0.get('MinTemp')
        x1c = wildfiredata0.get('AvgDewPoint')
        x1d = wildfiredata0.get('AvgWindSpeed')
        # x1e = wildfiredata0.get('AvgDayPrecip')

        x2a = wildfiredata1.get('MaxTemp')
        x2b = wildfiredata1.get('MinTemp')
        x2c = wildfiredata1.get('AvgDewPoint')
        x2d = wildfiredata1.get('AvgWindSpeed')
        # x2e = wildfiredata1.get('AvgDayPrecip')

        x1a.values(), x1b.values(), x1c.values(), x1d.values()# , x1e.values()
        x2a.values(), x2b.values(), x2c.values(), x2d.values()# , x2e.values()

        X_1a_untx = np.array(list(x1a.values()))
        X_1a_untx = X_1a_untx / np.max(X_1a_untx)
        X_1b_untx = np.array(list(x1b.values()))
        X_1b_untx = X_1b_untx / np.max(X_1b_untx) # 0.53 if not commented out
        X_1c_untx = np.array(list(x1c.values()))
        X_1c_untx = X_1c_untx / np.max(X_1c_untx)
        X_1d_untx = np.array(list(x1d.values()))
        X_1d_untx = X_1d_untx / np.max(X_1d_untx)
        # X_1e_untx = np.array(list(x1e.values()))
        # X_1e_untx = X_1d_untx / np.max(X_1e_untx)

        X_2a_untx = np.array(list(x2a.values()))
        X_2a_untx = X_2a_untx / np.max(X_2a_untx)
        X_2b_untx = np.array(list(x2b.values()))
        X_2b_untx = X_2b_untx / np.max(X_2b_untx)
        X_2c_untx = np.array(list(x2c.values()))
        X_2c_untx = X_2c_untx / np.max(X_2c_untx)
        X_2d_untx = np.array(list(x2d.values()))
        X_2d_untx = X_2d_untx / np.max(X_2d_untx)
        # X_2e_untx = np.array(list(x2e.values()))
        # X_2e_untx = X_2e_untx / np.max(X_2e_untx)

        X1a_txp = X_1a_untx.reshape(1999, 1)
        X1b_txp = X_1b_untx.reshape(1999, 1)
        X1c_txp = X_1c_untx.reshape(1999, 1)
        X1d_txp = X_1d_untx.reshape(1999, 1)
#      X1e_txp = X_1e_untx.reshape(1282, 1)

        X2a_txp = X_2a_untx.reshape(1999, 1)
        X2b_txp = X_2b_untx.reshape(1999, 1)
        X2c_txp = X_2c_untx.reshape(1999, 1)
        X2d_txp = X_2d_untx.reshape(1999, 1)
#     X2e_txp = X_2e_untx.reshape(1282, 1)

        X1 = np.hstack((X1a_txp, np.atleast_2d(X1b_txp), np.atleast_2d(X1c_txp), np.atleast_2d(X1d_txp)))
                   #      np.atleast_2d(X1e_txp)))
        X2 = np.hstack((X2a_txp, np.atleast_2d(X2b_txp), np.atleast_2d(X2c_txp), np.atleast_2d(X2d_txp)))
#                        np.atleast_2d(X2e_txp)))

        y1 = np.ones(len(x1a.values())) * -1
        y2 = np.ones(len(x2a.values()))
        return X1, y1, X2, y2


    def split_train(X1, y1, X2, y2):
        X1_train = X1[0:1600]
        y1_train = y1[0:1600]
        X2_train = X2[0:1600]
        y2_train = y2[0:1600]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train


    def split_test(X1, y1, X2, y2):
        X1_test = X1[1600:]
        y1_test = y1[1600:]
        X2_test = X2[1600:]
        y2_test = y2[1600:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test


    def wildfire_predict():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(gaussian_kernel)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))
        print("F1 score:", f1_score(y_test, y_predict, average='macro'))
        print(confusion_matrix(y_test, y_predict))
        print(classification_report(y_test, y_predict))


wildfire_predict()

