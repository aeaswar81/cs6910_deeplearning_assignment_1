This project uses Object Oriented Programming for modularity.
The core of this project is the class NeuralNetworkClassifier whose constructor can take the following parameters:
	1) no_of_neurons - A list which represents the architecture of the neural network. (i.e) number of neurons in each of the hidden layer
	2) n_class - Number of classes
	3) alpha - L2 regularization weight decay parameter. Default value is 0 which means no L2 regularization.
	4) activation - Activation function to be used. Default value is 'sigmoid'.
	5) output - Output function to be used. Default value is 'softmax'
	6) loss - Loss function to be used. Default value is 'cross_entropy'
	7) optimizer - Optimizer to be used. Default is None. Supported optimizers are sgd, momentum, nesterov, rmsprop, adam, nadam
	
This class has three useful methods namely fit, predict and accuracy.

1) fit(x, y, batch_size=BATCH_SIZE, epochs=100, eta=0.01, weight_initializer = None)
To train the neural network classifier.

x - training data
y - training labels
batch_size - batch size used for training
epochs - number of epochs until which the model will be trained
eta - learning rate
weight_initializer - weight initialization method to be used. By default it is None for which weights are initialized randomly. 'Xavier' method is supported.

Returns None

2) predict(X_test)
To predict the classes for given data

X_test - Data for which predictions are to be made

Returns an array of predictions

3) accuracy(X_test, y_test)
To calculate the accuracy of the model given some data

X_test - Data for which the accuracy of the model need to be evaluated
y_test - Corresponding class labels of the data


Example:
model = nn.NeuralNetworkClassifier([64, 64, 64], 10, alpha = 0.05, optimizer = 'adam')

model.fit(X_trainval, y_trainval, batch_size = 256, epochs = 150, eta = 0.001, 
          weight_initializer = 'Xavier')

y_predictions = model.predict(X_test)
acc = model.accuracy(X_test, y_test)

This example creates a neural network with 3 hidden layers each having 64 neurons with 10 output classes. The model is trained by calling the fit method.