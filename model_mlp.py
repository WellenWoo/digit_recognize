# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:40:32 2019
@author: WellenWoo
"""
from sklearn.neural_network import MLPClassifier
from model_svm import Preprocessor, Trainer, Tester
import time

class Trainer_nn(Trainer):
    def mlp(self, x_train, y_train):
        model = MLPClassifier(hidden_layer_sizes=(300, 300),
                    activation="relu", solver='sgd',
                    max_iter=100, early_stopping=True,
                    random_state=3)
        model.fit(x_train, y_train)
        return model

def run_train():
#    t0 = time.time()
    pt = Preprocessor()
    tr = Trainer_nn()
    
    X_train, y_train = pt.get_data_labels()
    X_test, y_test = pt.get_data_labels("test")
    
#    X_train, y_train = pt.load_data()
#    X_test, y_test = pt.load_data("mnist_test_data.npz")
    
    clf = tr.mlp(X_train, y_train)
    tr.save_model(clf, "mlp_mnist_Hu300x300ReluSgdIter100Acc96Sample60000.m")
    
    tester = Tester("mlp_mnist_Hu300x300ReluSgdIter100Acc96Sample60000.m")
    mt, score, repo = tester.clf_quality(X_test, y_test) 
    print(mt, score, repo)
    return clf