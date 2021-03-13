from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
import numpy as np
import NeuralNetwork as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns 


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
noOfHiddenLayers = 5
noOfneuronsEach = [128] * noOfHiddenLayers
inputNeuronSize = X_trainval.shape[1]

optimizers = ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']

optimizer = optimizers[5]
epochs = 40
eta = 0.0001
batch_size = 32
weight_initializer = 'Xavier'
alpha = 0 # weight decay rate for L2 regularization

model = nn.NeuralNetworkClassifier(noOfneuronsEach, l, alpha = alpha, activation = 'tanh', optimizer = optimizer)

model.fit(X_trainval, y_trainval, batch_size = batch_size, epochs = epochs, eta = eta, 
          weight_initializer = weight_initializer)

acc = model.accuracy(X_test, y_test)
print('Test Accuracy ', acc)

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

#%%
titles = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
y_pred=model.predict(X_test)
conf=confusion_matrix(y_test,y_pred)
print(conf)
#flatten the matrix 
numbers=conf.flatten()
#convert to percentage
percent=['{0:.2%}'.format(num) for num in numbers/np.sum(conf)]
p=plt.figure(figsize=(11,11))
#zip it 
values =[f'{num}\n{perc}' for num,perc in zip(numbers,percent)]
values=np.asarray(values).reshape(10,10)
sns.heatmap(conf,fmt='',annot=values,cmap='Blues',linewidths=2,linecolor="purple",square=True, xticklabels=titles, yticklabels=titles)
plt.ylabel("True class",size=20)
plt.xlabel('Predicted class',size=20)
plt.title("Confusion matrix heatmap",size=20)
#wandb.init()
plt.show()

#%%
wandb.init(project='cs6910-assignment1-sweep', name = 'confusion_matrix')
wandb.log({"Confusion matrix": [wandb.Image(p, caption="Confusion matrix")]})
wandb.finish()
