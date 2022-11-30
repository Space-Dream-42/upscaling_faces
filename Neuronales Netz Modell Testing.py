import tensorflow as tf
import numpy as np
import time
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

X = np.load('1.npy') #X values, X.shape = (768,1000) (16*16*3 Input features and 1000 trainingsexamples)
Y = np.load('88.npy') #Y values, Y.shape = (12288,1000) (64*64*3 Output features and 1000 trainingsexamples)
m = 1000   #Number of the trainingsexamples
alpha = 0.05 #Learning rate
iterations = 100
cache = 0 #Used to save our previous J to calculate if we are getting better or not

#Definiton of our NN modell

#First layer / first hidden layer
W_1 = np.random.randn(768,100) #weights
B_1 = np.zeros((100,1)) 
Z_1 = np.zeros((100,1000))
A_1 = np.zeros((100,1000)) #Activations of the first layer

#Second layer / second hidden layer
W_2 = np.random.randn(100,100)
B_2 = np.zeros((100,1))
Z_2 = np.zeros((100,1000))
A_2 = np.zeros((100,1000))

W_3 = np.random.randn(100,100)
B_3 = np.zeros((100,1))
Z_3 = np.zeros((100,1000))
A_3 = np.zeros((100,1000))

W_4 = np.random.randn(100,100)
B_4 = np.zeros((100,1))
Z_4 = np.zeros((100,1000))
A_4 = np.zeros((100,1000))

W_5 = np.random.randn(100,100)
B_5 = np.zeros((100,1))
Z_5 = np.zeros((100,1000))
A_5 = np.zeros((100,1000))

#Third layer/ output layer
W_6 = np.random.randn(100,12288)
B_6 = np.zeros((12288,1))
Z_6 = np.zeros((12288,1000))
A_6 = np.zeros((12288,1000))


for i in range(iterations):
    
    #Forwardprop start
    Z_1 = np.dot(W_1.T,X) + B_1
    A_1 = np.maximum(Z_1,0)

    Z_2 = np.dot(W_2.T,A_1) + B_2
    A_2 = np.maximum(Z_2,0)

    Z_3 = np.dot(W_3.T,A_2) + B_3
    A_3 = np.maximum(Z_3,0)

    Z_4 = np.dot(W_4.T,A_3) + B_4
    A_4 = np.maximum(Z_4,0)

    Z_5 = np.dot(W_5.T,A_4) + B_5
    A_5 = np.maximum(Z_5,0)

    Z_6 = np.dot(W_6.T,A_5) + B_6
    A_6 = np.maximum(Z_6, 0)

    #Calculates the loss and shows, if the NN is getting better or not
    Loss = np.square(Y - A_6)
    J = Loss.mean() 
    if J < cache:
        print('smalle')
    else:
        print('bigger')
    cache = J
    #Forwardprop End

    #BackProp Start
    da_dz = Z_6 >= 0 #da_dz = da/dz
    da_dz = da_dz.astype(np.int)
    dZ_6 = -2 * (Y - A_6) * da_dz #dZ_3 = dL(Y, A_3)/dZ_3
    dW_6 = np.dot(dZ_6,A_5.T) * (1/m) #dW_3 = dL(Y, A_3)/dW_3
    dB_6 = np.sum(dZ_6, axis = 1, keepdims = True) * (1/m) #dB_3 = dL(Y, A_3)/dB_3
  
    W_6 = W_6 - alpha * dW_6.T
    B_6 = B_6 - alpha * dB_6
    

    da_dz = Z_5 >= 0
    da_dz = da_dz.astype(np.int)
    dZ_5 = np.dot(dW_6.T, dZ_6) * da_dz    
    dW_5 = np.dot(dZ_5,A_4.T) * (1/m) 
    dB_5 = (np.sum(dZ_5, axis = 1, keepdims = True)) * (1/m) 

    W_5 = W_5 - alpha * dW_5.T
    B_5 = B_5 - alpha * dB_5


    da_dz = Z_4 >= 0
    da_dz = da_dz.astype(np.int)
    dZ_4 = np.dot(dW_5.T, dZ_5) * da_dz    
    dW_4 = np.dot(dZ_4,A_3.T) * (1/m) 
    dB_4 = (np.sum(dZ_4, axis = 1, keepdims = True)) * (1/m) 

    W_4 = W_4 - alpha * dW_4.T
    B_4 = B_4 - alpha * dB_4

    da_dz = Z_3 >= 0
    da_dz = da_dz.astype(np.int)
    dZ_3 = np.dot(dW_4.T, dZ_4) * da_dz    
    dW_3 = np.dot(dZ_3,A_2.T) * (1/m) 
    dB_3 = (np.sum(dZ_3, axis = 1, keepdims = True)) * (1/m) 

    W_3 = W_3 - alpha * dW_3.T
    B_3 = B_3 - alpha * dB_3

    da_dz = Z_2 >= 0
    da_dz = da_dz.astype(np.int)
    dZ_2 = np.dot(dW_3.T, dZ_3) * da_dz    
    dW_2 = np.dot(dZ_2,A_1.T) * (1/m) 
    dB_2 = (np.sum(dZ_2, axis = 1, keepdims = True)) * (1/m) 

    W_2 = W_2 - alpha * dW_2.T
    B_2 = B_2 - alpha * dB_2

    da_dz = Z_1 >= 0
    da_dz = da_dz.astype(np.int)
    dZ_1 = np.dot(dW_2.T, dZ_2) * da_dz    
    dW_1 = np.dot(dZ_1,X.T) * (1/m) 
    dB_1 = (np.sum(dZ_1, axis = 1, keepdims = True)) * (1/m)

    W_1 = W_1 - alpha * dW_1.T
    B_1 = B_1 - alpha * dB_1
    #BackProp End
    print(i)


