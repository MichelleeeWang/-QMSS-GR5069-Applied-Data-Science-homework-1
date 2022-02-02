import numpy as np
from numpy.linalg import inv
from pandas import read_csv
import matplotlib.pyplot as plt
# load dataset
X_train = read_csv('/Users/michelle/Desktop/Columbia_University_DS/COMS4721-Machine-Learning/hw1-data/X_train.csv', header=None)
y_train = read_csv('/Users/michelle/Desktop/Columbia_University_DS/COMS4721-Machine-Learning/hw1-data/y_train.csv', header=None)
X_test = read_csv('/Users/michelle/Desktop/Columbia_University_DS/COMS4721-Machine-Learning/hw1-data/X_test.csv', header=None)
y_test = read_csv('/Users/michelle/Desktop/Columbia_University_DS/COMS4721-Machine-Learning/hw1-data/y_test.csv', header=None)

def RidgeRegression(X, y):
    wrr_list = []
    dof_list = []
    for i in range(5001):
        lamda = i
        w_rr = (inv(X.transpose().dot(X) + i * np.identity(len(X[1,:])))).dot(X.transpose()).dot(y)
        U, s, VT = np.linalg.svd(X)
        dof = np.sum(np.square(s) / (np.square(s) + lamda))
        wrr_list.append(w_rr)
        dof_list.append(dof)
    return wrr_list, dof_list


def make_ridge_plot(wrr_list, dof_list):
    wrr_array = np.asarray(wrr_list)
    labels = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "year_made", "intercept"]
    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    for i in range(7):
        plt.plot(dof_list,wrr_array[:,i], label=labels[i])
    plt.legend(prop = {'size':15})
    plt.title("")
    plt.xlabel("Degree of freedom")
    plt.ylabel("Ridge regression parameter")



def get_RMSE_values(X_test, y_test, wrr_list, num_of_lamda,poly):
    RMSE_list = []
    wrr_array = np.asarray(wrr_list)
    for lamda in range(0, num_of_lamda+1):
        y_pred =  np.dot(np.array(X_test), wrr_array[lamda,:])
        RMSE = np.sqrt(np.sum(np.square(np.array(y_test) - y_pred))/len(y_pred))
        RMSE_list.append(RMSE)
    return RMSE_list


def plotRMSEValue(num_of_lamda, RMSE_list, poly):
    legends = ["1st order", "2nd order", "3rd order"]
    plt.rcParams['figure.figsize'] = (13.0, 8.0)
    lamda_list = list(range(num_of_lamda + 1))
    plt.plot(lamda_list, RMSE_list, label=legends[poly - 1])
    plt.legend(prop={'size': 15})
    plt.title("")
    plt.xlabel("lamda")
    plt.ylabel("RMSE")


def addPolynomialOrder(X_train, X_test, p):
    if p == 1:
        return X_train, X_test
    elif p == 2:
        a = X_train
        b = np.power(X_train[:, 0:6], 2)
        mean = b[:, 0:6].mean(axis=0)
        std = b[:, 0:6].std(axis=0)
        b = (b - mean) / std
        c = X_test
        d = np.power(X_test[:, 0:6], 2)
        d = (d - mean) / std
        X_train_new = np.hstack((a, b))
        X_test_new = np.hstack((c, d))
        return X_train_new, X_test_new
    elif p == 3:
        a = X_train
        b = np.power(X_train[:, 0:6], 2)
        c = np.power(X_train[:, 0:6], 3)
        mean_b = b.mean(axis=0)
        std_b = b[:, 0:6].std(axis=0)
        mean_c = c.mean(axis=0)
        std_c = c[:, 0:6].std(axis=0)
        b = (b - mean_b) / std_b
        c = (c - mean_c) / std_c
        a_test = X_test
        b_test = np.power(X_test[:, 0:6], 2)
        c_test = np.power(X_test[:, 0:6], 3)
        b_test = (b_test - mean_b) / std_b
        c_test = (c_test - mean_c) / std_c
        X_train_new = np.hstack((a, b, c))
        X_test_new = np.hstack((a_test, b_test, c_test))
        return X_train_new, X_test_new


##(a)
wrr_list, dof_list = RidgeRegression(np.array(X_train), y_train)
plt.figure()
make_ridge_plot(wrr_list, dof_list)
plt.show()
##(c)
RMSE_list = get_RMSE_values(X_test, y_test, wrr_list, 50,poly = 1)
plt.figure()
plotRMSEValue(50, RMSE_list,poly = 1)
plt.show()
##(d)
plt.figure()
for i in [1, 2, 3]:
    X_train_new, X_test_new = addPolynomialOrder(np.array(X_train), np.array(X_test), p=i)
    wrr_list, dof_list = RidgeRegression(X_train_new, y_train)
    wrrArray = np.asarray(wrr_list)
    dofArray = np.asarray(dof_list)
    RMSE_list = get_RMSE_values(X_test_new, y_test, wrrArray, 100, poly = i)
    plotRMSEValue(100, RMSE_list,poly = i)
plt.show()




