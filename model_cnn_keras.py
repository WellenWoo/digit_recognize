# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 19:36:25 2019
@author: WellenWoo
"""
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from utils import Preprocessor
import numpy as np

class Trainer(object):
    def cnn(self, X_train, y_train, X_test, y_test, 
            batch_size = 64, epochs = 12):
        input_shape = X_train.shape[1:]
        num_classes = y_train.shape[-1]
        
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation="relu", 
                         input_shape = input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), 
                         activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation="softmax"))
        
        model.compile(loss = "categorical_crossentropy",
                      optimizer = "adadelta",
                      metrics = ["accuracy"])
        
        model.fit(X_train, y_train, batch_size = batch_size,
                  epochs = epochs, 
                  verbose = 1, 
                  validation_data = (X_test, y_test))
        score = model.evaluate(X_test, y_test, verbose = 0)
        print("Test score:", score[0])
        print("Test accuracy:", score[1])
        return model

    def save_model(self,clf,output_name):
        """保存模型"""
        clf.save(output_name)

    def load_model(self,fn):
        """加载模型"""
        from keras.models import load_model
        clf = load_model(fn)
        return clf

    def plot_model(self,clf,output_name):
        """绘制神经网络"""
        from keras.utils import plot_model
        plot_model(clf,to_file = output_name,show_shapes = True,show_layer_names = True)

def run():
    pt = Preprocessor()    
    tr = Trainer()
    X_train, y_train = pt.load_data()
    X_test, y_test = pt.load_data("mnist_test_data.npz")
    
    x1 = X_train.reshape((-1, 28, 28, 1))
    x2 = X_test.reshape((-1, 28, 28, 1))
    
    y1 = keras.utils.to_categorical(y_train, len(np.unique(y_train)))
    y2 = keras.utils.to_categorical(y_test, len(np.unique(y_test)))
    
    clf = tr.cnn(x1, y1, x2, y2)
    tr.save(clf, "cnn_mnist_keras.h5")
    return clf