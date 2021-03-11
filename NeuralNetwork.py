# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 18:38:49 2021

@authors: Aravind Easwar, Arun Kumar
"""

import Functions as funs
import numpy as np

BATCH_SIZE = 600


class NeuralNetworkClassifier:
    def __init__(self, hidden_layers, hidden_units, n_class, activation='sigmoid',
                 output='softmax', output_units=1, loss='cross_entropy', optimizer=None):
        np.random.seed(13)
        self.hidden_layers = hidden_layers
        if len(hidden_units) != hidden_layers:
            raise Exception('Size of hidden_units array must be same as the number of hidden_layers')
        self.hidden_units = hidden_units
        self.n_class = n_class
        self.activation = activation
        self.output = output
        self.output_units = output_units
        self.loss = loss
        self.optimizer = optimizer
        self.is_trained = False
        self.activation_function = None
        self.activation_function_derivative = None
        self.init_activation_function()
        self.output_function = None
        self.output_function_derivative = None
        self.init_output_function()
        self.loss_function = None
        self.init_loss_function()
        self.parameters = {}
        self.loss_history = []
        self.val_loss_history = []
        self.accuracy_history = []
        self.val_accuracy_history = []

    def init_activation_function(self):
        if self.activation == 'sigmoid':
            self.activation_function = funs.sigmoid
            self.activation_function_derivative = funs.sigmoid_derivative
        elif self.activation == 'relu':
            self.activation_function = funs.relu
            self.activation_function_derivative = funs.relu_derivative
        elif self.activation == 'tanh':
            self.activation_function = funs.tanh
            self.activation_function_derivative = funs.tanh_derivate
        else:
            raise Exception("Unsupported activation function - ", self.activation)

    def init_output_function(self):
        if self.output == 'softmax':
            self.output_function = funs.softmax
            # TODO set output function derivative

    def init_loss_function(self):
        if self.loss == 'squared_loss':
            self.loss_function = funs.squared_loss
        elif self.loss == 'cross_entropy':
            self.loss_function = funs.cross_entropy

    def init_weights(self, input_size):
        self.parameters['W1'] = np.random.uniform(low=-0.5, high=0.5, size=(self.hidden_units[0], input_size))
        self.parameters['b1'] = np.random.uniform(low=-0.5, high=0.5, size=(self.hidden_units[0], 1))
        for i in range(2, self.hidden_layers + 1):
            self.parameters['W' + str(i)] = np.random.uniform(low=-0.5, high=0.5,
                                                              size=(self.hidden_units[i - 1], self.hidden_units[i - 2]))
            self.parameters['b' + str(i)] = np.random.uniform(low=-0.5, high=0.5, size=(self.hidden_units[i - 1], 1))
        # Output W
        self.parameters['W' + str(self.hidden_layers + 1)] = np.random.uniform(low=-0.5, high=0.5,
                                                                               size=(
                                                                                   self.n_class, self.hidden_units[-1]))
        self.parameters['b' + str(self.hidden_layers + 1)] = np.random.uniform(low=-0.5, high=0.5,
                                                                               size=(self.n_class, 1))

    def make_zeros_like(self, parameters):
        zero_parameters = {}
        for k in range(1, self.hidden_layers + 2):
            zero_parameters['W' + str(k)] = np.zeros_like(parameters['W' + str(k)])
            zero_parameters['b' + str(k)] = np.zeros_like(parameters['b' + str(k)])
        return zero_parameters

    def multiply_parameters(self, parameters, gamma):
        new_parameters = {}
        for k in range(1, self.hidden_layers + 2):
            new_parameters['W' + str(k)] = gamma * parameters['W' + str(k)]
            new_parameters['b' + str(k)] = gamma * parameters['b' + str(k)]
        return new_parameters

    def update_parameters(self, parameters, update, subtract=True):
        for i in range(1, self.hidden_layers + 2):
            if subtract:
                parameters['W' + str(i)] -= update['W' + str(i)]
                parameters['b' + str(i)] -= update['b' + str(i)]
            else:
                parameters['W' + str(i)] += update['W' + str(i)]
                parameters['b' + str(i)] += update['b' + str(i)]

    def forwardPropagation(self, X):
        preactivation = {}
        activation = {}
        activation['h0'] = X.T
        for k in range(1, self.hidden_layers + 1):
            preactivation['a' + str(k)] = np.dot(self.parameters['W' + str(k)], activation['h' + str(k - 1)]) + \
                                          self.parameters['b' + str(k)]
            activation['h' + str(k)] = self.activation_function(preactivation['a' + str(k)])
        preactivation['a' + str(self.hidden_layers + 1)] = np.dot(self.parameters['W' + str(self.hidden_layers + 1)],
                                                                  activation['h' + str(self.hidden_layers)]) + \
                                                           self.parameters['b' + str(
                                                               self.hidden_layers + 1)]
        y = funs.softmax(preactivation['a' + str(self.hidden_layers + 1)])
        return (preactivation, activation, y)

    def backPropagation(self, activation, preactivation, yhat, X, y_train):
        grads = {}
        eIndicator = np.zeros((self.n_class, X.shape[0]))
        eIndicator[y_train, np.arange(X.shape[0])] = 1
        grads['a' + str(self.hidden_layers + 1)] = -(eIndicator - yhat)
        for j in range(self.hidden_layers + 1, 0, -1):
            grads['W' + str(j)] = np.dot(grads['a' + str(j)], activation['h' + str(j - 1)].T)
            grads['b' + str(j)] = np.sum(grads['a' + str(j)], axis=1, keepdims=True)
            grads['h' + str(j - 1)] = np.dot(self.parameters['W' + str(j)].T, grads['a' + str(j)])
            if j != 1:
                grads['a' + str(j - 1)] = grads['h' + str(j - 1)] * self.activation_function_derivative(
                    preactivation['a' + str(j - 1)])
        return grads

    def update_loss_accuracy(self, X_train, X_val, y_train, y_val):
        _, _, yhat = self.forwardPropagation(X_train)
        _, _, yhatval = self.forwardPropagation(X_val)
        self.loss_history.append(self.loss_function(yhatval, y_val, X_val))
        self.val_loss_history.append(self.loss_function(yhat, y_train, X_train))
        self.accuracy_history.append(self.accuracy(X_val, y_val))
        self.val_accuracy_history.append(self.accuracy(X_train, y_train))

    def vanillaGradDescent(self, X_train, y_train, X_val, y_val, epochs, eta, n_examples, batchSize):
        t = 0
        while (t < epochs):
            print("Epoch ", t)
            mini = 0
            while (mini < (n_examples / batchSize)):
                # print("Batch ",mini)
                X_mini = X_train[(mini * batchSize):((mini + 1) * batchSize - 1)]
                y_mini = y_train[(mini * batchSize):((mini + 1) * batchSize - 1)]
                preactivation, activation, yhat = self.forwardPropagation(X_mini)
                gradients = self.backPropagation(activation, preactivation, yhat, X_mini, y_mini)
                for i in range(1, self.hidden_layers + 2):
                    self.parameters['W' + str(i)] -= eta * (1.0 / X_mini.shape[0]) * gradients['W' + str(i)]
                    self.parameters['b' + str(i)] -= eta * (1.0 / X_mini.shape[0]) * gradients['b' + str(i)]
                mini += 1
            t += 1
            self.update_loss_accuracy(X_train, X_val, y_train, y_val)

    def sgd(self, X_train, y_train, X_val, y_val, epochs, eta):
        t = 0
        while (t < epochs):
            print("iter ", t)
            # shuffle the data
            ids = np.random.permutation(len(X_train))
            X_random = X_train[ids]
            y_random = y_train[ids]
            for (x, y) in zip(X_random, y_random):
                batch_x = np.array([x])
                batch_y = np.array([y])
                preactivation, activation, yhat = self.forwardPropagation(batch_x)
                gradients = self.backPropagation(activation, preactivation, yhat, batch_x, batch_y)
                for i in range(1, self.hidden_layers + 2):
                    self.parameters['W' + str(i)] -= eta * gradients['W' + str(i)]
                    self.parameters['b' + str(i)] -= eta * gradients['b' + str(i)]
            t += 1
            self.update_loss_accuracy(X_train, X_val, y_train, y_val)

    def momentumGD(self, X_train, y_train, X_val, y_val, epochs, eta, n_examples, batchSize):
        # initialization
        prevW = {}
        prevb = {}
        gamma = 0.9
        prevW['W' + str(1)] = np.zeros((self.hidden_layers[0], X_train.shape[1]))
        prevb['b' + str(1)] = np.zeros((self.hidden_layers[0], 1))
        for i in range(2, self.hidden_layers + 1):
            prevW['W' + str(i)] = np.zeros((self.hidden_layers[i - 1], self.hidden_layers[i - 2]))
            prevb['b' + str(i)] = np.zeros((self.hidden_layers[i - 1], 1))
        prevW['W' + str(self.hidden_layers + 1)] = np.zeros((self.n_class, self.hidden_layers[-1]))
        prevb['b' + str(self.hidden_layers + 1)] = np.zeros((self.n_class, 1))
        t = 0
        while (t < epochs):
            print("Epoch ", t)
            mini = 0
            while (mini < (n_examples / batchSize)):
                # print("Batch ",mini)
                X_mini = X_train[(mini * batchSize):((mini + 1) * batchSize - 1)]
                y_mini = y_train[(mini * batchSize):((mini + 1) * batchSize - 1)]
                preactivation, activation, yhat = self.forwardPropagation(X_mini)
                gradients = self.backPropagation(activation, preactivation, yhat, X_mini, y_mini)
                # print("gradients",gradients)
                for i in range(1, self.hidden_layers + 2):
                    w = gamma * prevW['W' + str(i)] + eta * (1.0 / X_mini.shape[0]) * gradients['W' + str(i)]
                    b = gamma * prevb['b' + str(i)] + eta * (1.0 / X_mini.shape[0]) * gradients['b' + str(i)]
                    self.parameters['W' + str(i)] -= w
                    self.parameters['b' + str(i)] -= b
                    prevW['W' + str(i)] = w
                    prevb['b' + str(i)] = b
                mini += 1
            t += 1
            self.update_loss_accuracy(X_train, X_val, y_train, y_val)

    def nag(self, X_train, y_train, X_val, y_val, epochs, eta):
        t = 0
        prev_update = self.make_zeros_like(self.parameters)
        gamma = 0.9
        while (t < epochs):
            print("iter ", t)
            # do partial updates
            update = self.multiply_parameters(prev_update, gamma)
            self.update_parameters(self.parameters, update)
            preactivation, activation, yhat = self.forwardPropagation(X_train)
            gradients = self.backPropagation(activation, preactivation, yhat, X_train, y_train)
            eta_times_grad = self.multiply_parameters(gradients, eta * (1.0 / X_train.shape[0]))
            self.update_parameters(self.parameters, eta_times_grad)
            self.update_parameters(update, eta_times_grad, subtract=False)
            prev_update = update
            t += 1
            self.update_loss_accuracy(X_train, X_val, y_train, y_val)

    def rmsProp(self, X_train, y_train, X_val, y_val, epochs, eta, n_examples, batchSize):
        # initialization
        prevW = {}
        prevb = {}
        epsilon = 1e-8
        beta = 0.9
        prevW['W' + str(1)] = np.zeros((self.hidden_layers[0], X_train.shape[1]))
        prevb['b' + str(1)] = np.zeros((self.hidden_layers[0], 1))
        for i in range(2, self.hidden_layers + 1):
            prevW['W' + str(i)] = np.zeros((self.hidden_layers[i - 1], self.hidden_layers[i - 2]))
            prevb['b' + str(i)] = np.zeros((self.hidden_layers[i - 1], 1))
        prevW['W' + str(self.hidden_layers + 1)] = np.zeros((self.n_class, self.hidden_layers[-1]))
        prevb['b' + str(self.hidden_layers + 1)] = np.zeros((self.n_class, 1))
        t = 0
        while t < epochs:
            mini = 0
            while mini < (n_examples / batchSize):
                X_mini = X_train[(mini * batchSize):((mini + 1) * batchSize - 1)]
                y_mini = y_train[(mini * batchSize):((mini + 1) * batchSize - 1)]
                preactivation, activation, yhat = self.forwardPropagation(X_mini)
                gradients = self.backPropagation(activation, preactivation, yhat, X_mini, y_mini)
                for i in range(1, self.hidden_layers + 2):
                    prevW['W' + str(i)] = beta * prevW['W' + str(i)] + (1.0 - beta) * (gradients['W' + str(i)] ** 2)
                    prevb['b' + str(i)] = beta * prevb['b' + str(i)] + (1.0 - beta) * (gradients['b' + str(i)] ** 2)

                    self.parameters['W' + str(i)] -= (
                            (eta / np.sqrt(prevW['W' + str(i)] + epsilon)) * gradients['W' + str(i)])
                    self.parameters['b' + str(i)] -= (
                            (eta / np.sqrt(prevb['b' + str(i)] + epsilon)) * gradients['b' + str(i)])
                mini += 1
            t += 1
            self.update_loss_accuracy(X_train, X_val, y_train, y_val)

    def adam(self, X_train, y_train, X_val, y_val, epochs, eta, n_examples, batchSize):
        prevW = {}
        prevb = {}
        mW = {}
        mb = {}
        epsilon = 1e-8
        beta1 = 0.9
        beta2 = 0.999
        prevW['W' + str(1)] = np.zeros((self.hidden_layers[0], X_train.shape[1]))
        prevb['b' + str(1)] = np.zeros((self.hidden_layers[0], 1))
        for i in range(2, self.hidden_layers + 1):
            prevW['W' + str(i)] = np.zeros((self.hidden_layers[i - 1], self.hidden_layers[i - 2]))
            prevb['b' + str(i)] = np.zeros((self.hidden_layers[i - 1], 1))
        prevW['W' + str(self.hidden_layers + 1)] = np.zeros((self.n_class, self.hidden_layers[-1]))
        prevb['b' + str(self.hidden_layers + 1)] = np.zeros((self.n_class, 1))

        mW['W' + str(1)] = np.zeros((self.hidden_layers[0], X_train.shape[1]))
        mb['b' + str(1)] = np.zeros((self.hidden_layers[0], 1))
        for i in range(2, self.hidden_layers + 1):
            mW['W' + str(i)] = np.zeros((self.hidden_layers[i - 1], self.hidden_layers[i - 2]))
            mb['b' + str(i)] = np.zeros((self.hidden_layers[i - 1], 1))
        mW['W' + str(self.hidden_layers + 1)] = np.zeros((self.n_class, self.hidden_layers[-1]))
        mb['b' + str(self.hidden_layers + 1)] = np.zeros((self.n_class, 1))

        t = 0  # iterations
        f = 0  # update number
        while t < epochs:
            print("Epoch ", t)
            mini = 0
            while mini < (n_examples / batchSize):
                X_mini = X_train[(mini * batchSize):((mini + 1) * batchSize - 1)]
                y_mini = y_train[(mini * batchSize):((mini + 1) * batchSize - 1)]
                preactivation, activation, yhat = self.forwardPropagation(X_mini)
                gradients = self.backPropagation(activation, preactivation, yhat, X_mini, y_mini)
                f += 1
                for i in range(1, self.hidden_layers + 2):
                    mW['W' + str(i)] = beta1 * mW['W' + str(i)] + (1.0 - beta1) * (gradients['W' + str(i)])
                    mb['b' + str(i)] = beta1 * mb['b' + str(i)] + (1.0 - beta1) * (gradients['b' + str(i)])

                    prevW['W' + str(i)] = beta2 * prevW['W' + str(i)] + (1.0 - beta2) * (gradients['W' + str(i)] ** 2)
                    prevb['b' + str(i)] = beta2 * prevb['b' + str(i)] + (1.0 - beta2) * (gradients['b' + str(i)] ** 2)

                    mWHat = (1.0 - (beta1 ** f)) * mW['W' + str(i)]
                    mbHat = (1.0 - (beta1 ** f)) * mb['b' + str(i)]

                    vWHat = (1.0 - (beta2 ** f)) * prevW['W' + str(i)]
                    vbHat = (1.0 - (beta2 ** f)) * prevb['b' + str(i)]

                    self.parameters['W' + str(i)] -= ((eta / np.sqrt(vWHat + epsilon)) * mWHat)
                    self.parameters['b' + str(i)] -= ((eta / np.sqrt(vbHat + epsilon)) * mbHat)
                mini += 1
            t += 1
            self.update_loss_accuracy(X_train, X_val, y_train, y_val)

    def nadam(self, X_train, y_train, X_val, y_val, epochs, eta, n_examples, batchSize):
        # initialization
        prevW = {}
        prevb = {}
        mW = {}
        mb = {}
        epsilon = 1e-8
        beta1 = 0.9
        beta2 = 0.999
        prevW['W' + str(1)] = np.zeros((self.hidden_layers[0], X_train.shape[1]))
        prevb['b' + str(1)] = np.zeros((self.hidden_layers[0], 1))
        for i in range(2, self.hidden_layers + 1):
            prevW['W' + str(i)] = np.zeros((self.hidden_layers[i - 1], self.hidden_layers[i - 2]))
            prevb['b' + str(i)] = np.zeros((self.hidden_layers[i - 1], 1))
        prevW['W' + str(self.hidden_layers + 1)] = np.zeros((self.n_class, self.hidden_layers[-1]))
        prevb['b' + str(self.hidden_layers + 1)] = np.zeros((self.n_class, 1))

        mW['W' + str(1)] = np.zeros((self.hidden_layers[0], X_train.shape[1]))
        mb['b' + str(1)] = np.zeros((self.hidden_layers[0], 1))
        for i in range(2, self.hidden_layers + 1):
            mW['W' + str(i)] = np.zeros((self.hidden_layers[i - 1], self.hidden_layers[i - 2]))
            mb['b' + str(i)] = np.zeros((self.hidden_layers[i - 1], 1))
        mW['W' + str(self.hidden_layers + 1)] = np.zeros((self.n_class, self.hidden_layers[-1]))
        mb['b' + str(self.hidden_layers + 1)] = np.zeros((self.n_class, 1))

        t = 0  # iterations
        f = 0  # update number
        while t < epochs:
            print("Epoch ", t)
            mini = 0
            while mini < (n_examples / batchSize):
                X_mini = X_train[(mini * batchSize):((mini + 1) * batchSize - 1)]
                y_mini = y_train[(mini * batchSize):((mini + 1) * batchSize - 1)]
                preactivation, activation, yhat = self.forwardPropagation(X_mini)
                gradients = self.backPropagation(activation, preactivation, yhat, X_mini, y_mini)
                f += 1
                for i in range(1, self.hidden_layers + 2):
                    mW['W' + str(i)] = beta1 * mW['W' + str(i)] + (1.0 - beta1) * (gradients['W' + str(i)])
                    mb['b' + str(i)] = beta1 * mb['b' + str(i)] + (1.0 - beta1) * (gradients['b' + str(i)])

                    prevW['W' + str(i)] = beta2 * prevW['W' + str(i)] + (1.0 - beta2) * (gradients['W' + str(i)] ** 2)
                    prevb['b' + str(i)] = beta2 * prevb['b' + str(i)] + (1.0 - beta2) * (gradients['b' + str(i)] ** 2)

                    mWHat = (1.0 - (beta1 ** f)) * mW['W' + str(i)]
                    mbHat = (1.0 - (beta1 ** f)) * mb['b' + str(i)]

                    vWHat = (1.0 - (beta2 ** f)) * prevW['W' + str(i)]
                    vbHat = (1.0 - (beta2 ** f)) * prevb['b' + str(i)]

                    mbarW = beta1 * mWHat + (1.0 - beta1) * gradients['W' + str(i)]
                    mbarb = beta1 * mbHat + (1.0 - beta1) * gradients['b' + str(i)]

                    self.parameters['W' + str(i)] -= ((eta / np.sqrt(vWHat + epsilon)) * mbarW)
                    self.parameters['b' + str(i)] -= ((eta / np.sqrt(vbHat + epsilon)) * mbarb)
                mini += 1
            t += 1
            self.update_loss_accuracy(X_train, X_val, y_train, y_val)

    def fit(self, x, y, batch_size=BATCH_SIZE, epochs=100, eta=0.01):
        # momentum, nesterov, rmsprop, adam, nadam
        # initialize all parameters (weights and bias)
        self.init_weights(x.shape[1])
        # split data into train and validation set
        indices = np.random.permutation(x.shape[0])
        training_idx = indices[:54000]
        validation_idx = indices[54000:]
        x_train = x[training_idx, :]
        x_val = x[validation_idx, :]
        y_train = y[training_idx, :]
        y_val = y[validation_idx, :]
        if self.optimizer == 'sgd':
            self.sgd(x_train, y_train, x_val, y_val, epochs, eta)
        elif self.optimizer == 'momentum':
            self.momentumGD(x_train, y_train, x_val, y_val, epochs, eta, x_train.shape[0], batch_size)
        elif self.optimizer == 'nesterov':
            self.nag(x_train, y_train, x_val, y_val, epochs, eta)
        elif self.optimizer == 'rmsprop':
            self.rmsProp(x_train, y_train, x_val, y_val, epochs, eta, x_train.shape[0], batch_size)
        elif self.optimizer == 'adam':
            self.adam(x_train, y_train, x_val, y_val, epochs, eta, x_train.shape[0], batch_size)
        elif self.optimizer == 'nadam':
            self.nadam(x_train, y_train, x_val, y_val, epochs, eta, x_train.shape[0], batch_size)
        elif self.optimizer is None:
            self.vanillaGradDescent(x_train, y_train, x_val, y_val, epochs, eta, x_train.shape[0], batch_size)
        else:
            raise Exception("Unsupported optimizer.")

    def predict(self, X_test, y_test):
        if not self.is_trained:
            print('Network is not trained. Fit has to be invoked before calling predict')
            return None
        # do a forward pass
        _, _, y_hat = self.forwardPropagation(X_test)
        y_hat = y_hat.argmax(axis=0)
        return y_hat

    def accuracy(self, X_test, y_test):
        y_hat = self.predict(X_test, y_test)
        correctPred = np.sum(y_hat == y_test)
        return (correctPred / X_test.shape[0]) * 100
