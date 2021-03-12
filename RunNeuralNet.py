from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
import numpy as np
import NeuralNetwork as nn

#%% Load data

# load the data into train and test
(X_trainval, y_trainval), (X_test, y_test) = fashion_mnist.load_data()

n_examples = X_trainval.shape[0]
print(n_examples)
X_trainval = (1.0 / 255) * np.array([X_trainval[i].flatten() for i in range(0, X_trainval.shape[0])])
print(X_trainval.shape)
X_test = (1.0 / 255) * np.array([X_test[i].flatten() for i in range(0, X_test.shape[0])])

print(X_test.shape)
print(X_trainval.shape)

#%% Train model
l = 10  # output classes
noOfneuronsEach = [64, 64]
noOfHiddenLayers = len(noOfneuronsEach)
inputNeuronSize = X_trainval.shape[1]

optimizers = ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']

optimizer = optimizers[5]
epochs = 150
eta = 0.001
weight_initializer = 'Xavier'

model = nn.NeuralNetworkClassifier(noOfneuronsEach, l, optimizer = optimizer)

model.fit(X_trainval, y_trainval, epochs = epochs, eta = eta, weight_initializer = weight_initializer)

acc = model.accuracy(X_test, y_test)
print('Accuracy ', acc)

#%%
plt.figure(0)
plt.plot(model.loss_history)
plt.plot(model.val_loss_history)
title = '%s, %s, epochs = %d, eta = %f' % (str(noOfneuronsEach), optimizer, epochs, eta)
plt.title(title)
plt.legend(['training loss', 'validation loss'])
plt.show()

plt.figure(1)
plt.plot(model.accuracy_history)
plt.plot(model.val_accuracy_history)
plt.title(title)
plt.legend(['training accuracy', 'validation accuracy'])
plt.show()