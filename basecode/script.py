import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    features_new = n_features + 1
    w = initialWeights.reshape((features_new,1))
    bias = np.ones((n_data,1))
#     print(bias)
#     print(w)  
    x = np.hstack((bias,train_data))
#     print(x)
    z = np.dot(x,w)
    theta = sigmoid(z)
#     print("sigmoid",theta) 
    first = labeli * np.log(theta)
    second = (1.0 - labeli) * np.log(1.0 - theta)
    numerator = -np.sum(first + second)
    denominator = n_data
    error = numerator / denominator
#     print("Error value ",error)
    error_grad = (theta - labeli) * x
    error_grad = np.sum(error_grad, axis=0) / n_data
#   print("error_grad",error_grad)
    return error, error_grad

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    bias = np.ones((data.shape[0],1))
    d = np.hstack((bias,data))
    z = np.dot(d,W)
    label = sigmoid(z)
    label = np.argmax(label, axis = 1) # Each class - Maximum
    label = label.reshape((data.shape[0], 1))
    

    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.
    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector
    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    bias = np.ones((n_data,1))
    x = np.hstack((bias,train_data))
    features_new = n_feature + 1
    w = params.reshape((features_new,n_class))
    prod = np.dot(x , w)
    numerator = np.exp(prod)
    denominator = np.sum((np.exp(prod)),axis = 1).reshape(n_data , 1)
    theta = numerator / denominator
    y = labeli * np.log(theta)
    error = (-1 * np.sum(np.sum(y))) / n_data
    error_grad = (np.dot(x.T, (theta - labeli))) / n_data
    
    return error, error_grad.flatten()

def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    size = data.shape[0]
    bias = np.ones((size,1))
    x = np.hstack((bias,data))
    prod = np.dot(x,W)
    num = np.exp(prod)
    denom = np.sum(np.exp(prod))
    label = num / denom
    label = np.argmax(label, axis=1)
    label = label.reshape((size,1))

    return label

"""
Script for Logistic Regression
"""
start_time_LR = time.time()
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
print(confusion_matrix(train_label, predicted_label, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]))

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
print(confusion_matrix(validation_label, predicted_label, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])) 

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
print(confusion_matrix(test_label, predicted_label, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
stop_time_LR = time.time() - start_time_LR
print("Time taken for Logistic Regression {}.seconds\n".format(str(stop_time_LR)))

# Code for SVM
print("Learning SVM Using Linear Kernel")

svm = SVC(kernel = 'linear')
#train_label = train_label.flatten()
indexes = np.random.randint(50000, size = 10000)
sample_data = train_data[indexes, :]
sample_label = train_label[indexes, :]
svm.fit(sample_data, sample_label.flatten())

traning_accuracy = svm.score(train_data, train_label)
traning_accuracy = str(100*traning_accuracy)
print("Traning data Accuracy for Linear Kernel: {}%\n".format(traning_accuracy))
validation_accuracy = svm.score(validation_data, validation_label)
validation_accuracy = str(100*validation_accuracy)
print("Validation data Accuracy for Linear Kernel: {}%\n".format(validation_accuracy))
test_accuracy = svm.score(test_data, test_label)
test_accuracy = str(100*test_accuracy)
print("Test data Accuracy for Linear Kernel: {}%\n".format(test_accuracy))
time_linear_kernel = time.time() - start_time_linear_kernel

print("Time taken for SVM using Linear Kernel {}.seconds\n\n\n".format(str(time_linear_kernel)))


print("SVM with radial basis function with value of gamma setting to 1 ")
start_time_rbf = time.time()
svm = SVC(kernel = 'rbf', gamma = 1.0)
#train_label = train_label.flatten()
indexes = np.random.randint(50000, size = 10000)
sample_data = train_data[indexes, :]
sample_label = train_label[indexes, :]
svm.fit(sample_data, sample_label.flatten())
traning_accuracy_rbf = svm.score(train_data, train_label)
traning_accuracy_rbf = str(100*traning_accuracy_rbf)
print("Traning data Accuracy for rbf Kernel: {}%\n".format(traning_accuracy_rbf))
validation_accuracy_rbf = svm.score(validation_data, validation_label)
validation_accuracy_rbf= str(100*validation_accuracy_rbf)
print("Validation data Accuracy for rbf Kernel: {}%\n".format(validation_accuracy_rbf))
test_accuracy_rbf = svm.score(test_data, test_label)
test_accuracy_rbf = str(100*test_accuracy_rbf)
print("Test data Accuracy for rbf Kernel: {}%\n".format(test_accuracy_rbf))

time_rbf = time.time() - start_time_rbf
print("Time taken for SVM using rbf {}seconds\n\n\n".format(str(time_rbf)))

######
print("SVM with radial basis function with default gamma ")
start_time_rbf = time.time()
svm = SVC(kernel = 'rbf', gamma = "auto")
#train_label = train_label.flatten()
indexes = np.random.randint(50000, size = 10000)
sample_data = train_data[indexes, :]
sample_label = train_label[indexes, :]
svm.fit(sample_data, sample_label.flatten())
traning_accuracy_rbf = svm.score(train_data, train_label)
traning_accuracy_rbf = str(100*traning_accuracy_rbf)
print("Traning data Accuracy for rbf Kernel: {}%\n".format(traning_accuracy_rbf))
validation_accuracy_rbf = svm.score(validation_data, validation_label)
validation_accuracy_rbf= str(100*validation_accuracy_rbf)
print("Validation data Accuracy for rbf Kernel: {}%\n".format(validation_accuracy_rbf))
test_accuracy_rbf = svm.score(test_data, test_label)
test_accuracy_rbf = str(100*test_accuracy_rbf)
print("Test data Accuracy for rbf Kernel: {}%\n".format(test_accuracy_rbf))

time_rbf = time.time() - start_time_rbf
print("Time taken for SVM using rbf and default gamma {}seconds\n\n\n".format(str(time_rbf)))



print(" SVM with radial basis function with value of gamma setting to default and varying value of C")
start_time_varing_C = time.time()
C_val = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

traning_accuracy_C_val = np.zeros(11)
validation_accuracy_C_val = np.zeros(11)
test_accuracy_C_val = np.zeros(11)

for i in range(0, len(C_val)):
    print("Computing for C = {}".format(C_val[i]))
    svm = SVC(C = C_val[i], kernel = 'rbf', gamma = 'auto')
    #train_label = train_label.flatten()
    indexes = np.random.randint(50000, size = 10000)
    sample_data = train_data[indexes, :]
    sample_label = train_label[indexes, :]
    svm.fit(sample_data, sample_label.flatten())
    traning_accuracy_C_val[i] = svm.score(train_data, train_label)
    traning_accuracy_C_val[i] = str(100 * traning_accuracy_C_val[i])
    print("Traning data Accuracy: {}%\n".format(traning_accuracy_C_val[i]))
    validation_accuracy_C_val[i] = svm.score(validation_data, validation_label)
    validation_accuracy_C_val[i] = 100 * validation_accuracy_C_val[i]
    print("Validation data Accuracy: {}%\n".format(str(validation_accuracy_C_val[i])))
    test_accuracy_C_val[i] = svm.score(test_data, test_label)
    test_accuracy_C_val[i] = 100 * test_accuracy_C_val[i]
    print("Test data Accuracy: {}%\n".format(str(test_accuracy_C_val[i])))

    time_varing_C = time.time() - start_time_varing_C
     print("Time taken for SVM with radial basis function and varing C is {}.seconds\n\n\n".format(str(time_varing_C)))
    
    
print("Computing Optimal SVM on whole training dataset with radial basis function kernal and value of gamma setting to default and C = 50 ")
start_time_rbf = time.time()
svm = SVC(kernel = 'rbf', gamma = "auto", C = 50)
#train_label = train_label.flatten()
svm.fit(train_data, train_label.flatten())
traning_accuracy_rbf = svm.score(train_data, train_label)
traning_accuracy_rbf = str(100*traning_accuracy_rbf)
print("Traning data Accuracy for rbf Kernel: {}%\n".format(traning_accuracy_rbf))
validation_accuracy_rbf = svm.score(validation_data, validation_label)
validation_accuracy_rbf= str(100*validation_accuracy_rbf)
print("Validation data Accuracy for rbf Kernel: {}%\n".format(validation_accuracy_rbf))
test_accuracy_rbf = svm.score(test_data, test_label)
test_accuracy_rbf = str(100*test_accuracy_rbf)
print("Test data Accuracy for rbf Kernel: {}%\n".format(test_accuracy_rbf))

time_rbf = time.time() - start_time_rbf
print("Time taken for SVM with best choice of parameters {}seconds\n\n\n".format(str(time_rbf)))


# FOR EXTRA CREDIT ONLY
start = time.time()
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
print(confusion_matrix(train_label, predicted_label_b, labels = [0,1,2,3,4,5,6,7,8,9]))

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')
print(confusion_matrix(validation_label, predicted_label_b, labels = [0,1,2,3,4,5,6,7,8,9]))

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
print(confusion_matrix(test_label, predicted_label_b, labels = [0,1,2,3,4,5,6,7,8,9]))
end_time = time.time() - start
print("Time taken :{} .sec".format(str(end_time)))