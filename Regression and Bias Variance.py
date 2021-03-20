#!/usr/bin/env python
# coding: utf-8

# In[573]:


import numpy as np 
import math
import random as rand
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
x=np.random.rand(100)
noise=np.random.normal(0,np.sqrt(0.2),100)
y = np.zeros(len(x))
for i in range(100):
    y[i]=np.exp(math.sin(2*np.pi*x[i]))+x[i] + noise[i]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=1)

def func_true(x):
    return np.exp(math.sin(2*np.pi*x)) + x
def rms_error(y_crrct, y_pred):
    return np.sqrt(np.mean((y_crrct - y_pred)**2))
x_target_plot = np.linspace(0, 1, 1e5)
y_target_plot = np.vectorize(func_true)(x_target_plot)
train_rms_error=np.zeros(4)
test_rms_error=np.zeros(4)
plt.plot(x_target_plot, y_target_plot)
plt.scatter(x, y, marker='*')
plt.show()
train_rmse = []
test_rmse = []
degree_array = [1, 3, 6, 9]
degree = 1
X_1_plot = np.vstack([x_target_plot**degree, x_target_plot**0]).T
X = np.vstack([x_train**degree, x_train**0]).T
X_1_test = np.vstack([x_test**degree, x_test**0]).T

w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y_train)

y_pred_train = np.dot(X, w)
y_plot = np.dot(X_1_plot, w)
y_pred_test = np.dot(X_1_test, w)
plt.scatter(x_train, y_train, marker='.', c='g', label="Train Data")
plt.scatter(x_test, y_test, marker='.', c='black', label="Test Data")
plt.plot(x_target_plot, y_target_plot, c='orange', label="Target Func")
plt.plot(x_target_plot, y_plot, c='blue', label="Estimated Func")
plt.legend()
plt.title('Polynomial Regression for degree 1')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
train_rmse.append(rms_error(y_train, y_pred_train))
test_rmse.append(rms_error(y_test, y_pred_test))
print("Test Error: ", rms_error(y_test, y_pred_test))
degree = 3
X_3_plot = np.vstack([x_target_plot**degree, x_target_plot**2, x_target_plot, x_target_plot**0]).T
X = np.vstack([x_train**degree, x_train**2, x_train, x_train**0]).T
X_3_test = np.vstack([x_test**degree, x_test**2, x_test, x_test**0]).T

w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y_train)

y_pred_train = np.dot(X, w)
y_plot = np.dot(X_3_plot, w)
y_pred_test = np.dot(X_3_test, w)
plt.scatter(x_train, y_train, marker='.', c='g', label="Train Data")
plt.scatter(x_test, y_test, marker='.', c='black', label="Test Data")
plt.plot(x_target_plot, y_target_plot, c='orange', label="Target Func")
plt.plot(x_target_plot, y_plot, c='blue', label="Estimated Func")
plt.legend()
plt.title('Polynomial Regression for degree 3')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
plt.scatter(y_test, y_pred_test)
plt.plot(np.linspace(0, 4, 100), np.linspace(0, 4, 100), c='g')
plt.show()
plt.title("Training estimate Vs Training target for degree 3")
plt.xlabel("Training target")
plt.ylabel("Training estimate")
plt.plot(rms_error(y_train, y_pred_train))
print("RMS Error Train", rms_error(y_train, y_pred_train))

train_rmse.append(rms_error(y_train, y_pred_train))
test_rmse.append(rms_error(y_test, y_pred_test))
degree = 6
X_6_plot = np.vstack([x_target_plot**degree,x_target_plot**5,x_target_plot**4,x_target_plot**3,x_target_plot**2,x_target_plot, x_target_plot**0]).T
X = np.vstack([x_train**degree, x_train**5, x_train**4, x_train**3, x_train**2, x_train, x_train**0]).T
X_6_test = np.vstack([x_test**degree,x_test**5,x_test**4,x_test**3,x_test**2, x_test, x_test**0]).T

w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y_train)

y_pred_train = np.dot(X, w)
y_plot = np.dot(X_6_plot, w)
y_pred_test = np.dot(X_6_test, w)
plt.scatter(x_train, y_train, marker='.', c='g', label="Train Data")
plt.scatter(x_test, y_test, marker='.', c='black', label="Test Data")
plt.plot(x_target_plot, y_target_plot, c='orange', label="Target Func")
plt.plot(x_target_plot, y_plot, c='blue', label="Estimated Func")
plt.legend()
plt.title('Polynomial Regression for degree 6')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

train_rmse.append(rms_error(y_train, y_pred_train))
test_rmse.append(rms_error(y_test, y_pred_test))
degree = 9
X_9_plot = np.vstack([x_target_plot**degree,x_target_plot**8,x_target_plot**7,x_target_plot**6,x_target_plot**5,x_target_plot**4,x_target_plot**3,x_target_plot**2,x_target_plot, x_target_plot**0]).T
X = np.vstack([x_train**degree,x_train**8,x_train**7,x_train**6,x_train**5, x_train**4, x_train**3, x_train**2, x_train, x_train**0]).T
X_9_test = np.vstack([x_test**degree,x_test**8,x_test**7,x_test**6,x_test**5,x_test**4,x_test**3,x_test**2, x_test, x_test**0]).T

w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y_train)

y_pred_train = np.dot(X, w)
y_plot = np.dot(X_9_plot, w)
y_pred_test = np.dot(X_9_test, w)
plt.scatter(x_train, y_train, marker='.', c='g', label="Train Data")
plt.scatter(x_test, y_test, marker='.', c='black', label="Test Data")
plt.plot(x_target_plot, y_target_plot, c='orange', label="Target Func")
plt.plot(x_target_plot, y_plot, c='blue', label="Estimated Func")
plt.legend()
plt.title('Polynomial Regression for degree 9')
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

train_rmse.append(rms_error(y_train, y_pred_train))
test_rmse.append(rms_error(y_test, y_pred_test))
plt.plot(degree_array, train_rmse, label="Train RMSE")
plt.plot(degree_array, test_rmse, label="Test RMSE")
plt.legend()
plt.ylim(0, 10)
plt.show()


# In[ ]:




