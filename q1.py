# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 11:57:09 2021

@author: Arun-PC
"""

import matplotlib.pyplot as plt 
from keras.datasets import fashion_mnist
import wandb

# 1. Start a W&B run
wandb.init(project='cs6910-assignment1-sweep', name = 'class-samples-1')



# Function to plot images
def plotGallery(images, titles, h, w, n_row = 2, n_col = 5):
    fig = plt.figure(figsize =(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom = 0, left =.01, right =.99, top =.90, hspace =.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap = plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    return fig
        
def getSampleForClass(x, y, c):
    for i,l in zip(x, y):
        if l == c:
            return i        

(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
samples = []
for c in range(0,10):
    s = getSampleForClass(trainX, trainY, c)
    samples.append(s)

titles = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fig = plotGallery(samples, titles, 28, 28)

wandb.log({"examples": wandb.Image(fig)})
wandb.finish()