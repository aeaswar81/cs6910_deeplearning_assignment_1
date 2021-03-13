# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 20:10:14 2021

@author: Arun-PC
"""


import wandb
from keras.datasets import fashion_mnist
import numpy as np
import NeuralNetwork as nn
from matplotlib import pyplot as plt
#%%
# load the data into train and test
(X_trainval, y_trainval), (X_test, y_test) = fashion_mnist.load_data()

n_examples = X_trainval.shape[0]
print(n_examples)
X_trainval = (1.0 / 255) * np.array([X_trainval[i].flatten() for i in range(0, X_trainval.shape[0])])
print(X_trainval.shape)
X_test = (1.0 / 255) * np.array([X_test[i].flatten() for i in range(0, X_test.shape[0])])

print(X_test.shape)
print(X_trainval.shape)

#%%

def sweep():
    wandb.init()
    config = wandb.config
    name = "hl_"+str(config.hidden_layers)+"_bs_"+str(config.batch_size)+"_ac_"+str(config.activation)
    wandb.run.name = name
    
    noOfneuronsEach = [config.size_layer] * config.hidden_layers    
    model = nn.NeuralNetworkClassifier(noOfneuronsEach, 10, alpha = config.weight_decay,
                                       activation = config.activation , optimizer = config.optimizer)

    model.fit(X_trainval, y_trainval, batch_size = config.batch_size, epochs = config.epochs, eta = config.learning_rate, 
              weight_initializer = config.weight_init)
    
    acc = model.accuracy(X_test, y_test)
    print('test accuracy ', acc)
    wandb.log({'test_accuracy' : acc})
    t = 1
    for (val_loss, loss, acc, val_acc) in zip(model.val_loss_history, model.loss_history, model.accuracy_history, model.val_accuracy_history):
        wandb.log({'val_loss':val_loss, 'loss':loss, 'val_accuracy':val_acc ,'accuracy':acc, 'epochs': t})
        t += 1
        
    # plt.figure(0)
    # plt.plot(model.loss_history)
    # plt.plot(model.val_loss_history)
    # title = '%s, %s, epochs = %d, eta = %f' % (str(noOfneuronsEach), config.optimizer, config.epochs, config.learning_rate)
    # plt.title(title)
    # plt.legend(['training loss', 'validation loss'])
    # plt.show()
    
    # plt.figure(1)
    # plt.plot(model.accuracy_history)
    # plt.plot(model.val_accuracy_history)
    # plt.title(title)
    # plt.legend(['training accuracy', 'validation accuracy'])
    # plt.show()

#%%
    
sweep_config = {
    'method': 'bayes', #grid, random
    'metric': {
      'name': 'val_loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'epochs': {
            'values': [10,20,40]
        },
        'hidden_layers': {
            'values': [3,4,5]
        },
        'size_layer': {
            'values': [32, 64, 128]
        },
        'weight_decay': {
            'values': [0,0.0005, 0.5]
        },
        'learning_rate': {
            'values': [ 1e-2, 1e-3, 1e-4]
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
        },
        'batch_size':{
            'values': [16,32,64]
        },
        'weight_init':{
            'values': ['random','Xavier']
        },
        'activation': {
            'values': ['tanh','sigmoid', 'relu']
        }
    }
}
    
# wandb.init(project='cs6910-assignment1', name = 'class-samples-1')
sweep_id = wandb.sweep(sweep_config, project = "cs6910-assignment1-sweep")

#%% start wandb sweep
wandb.agent(sweep_id, sweep, count = 120)