# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:16:16 2019
@author: WellenWoo
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report
import time
from utils import Preprocessor

class Trainer(object):
    '''训练器;'''
    def svc(self, x_train, y_train):
        '''构建分类器'''
        model = SVC(kernel = 'poly',degree = 4,probability= True)
        model.fit(x_train, y_train)
        return model
        
    def save_model(self, model, output_name):
        '''保存模型'''
        joblib.dump(model,output_name, compress = 1)

    def load_model(self, model_path):
        '''加载模型'''
        clf = joblib.load(model_path)
        return clf

class Tester(object):
    '''测试器;'''
    def __init__(self, model_path):
        tr = Trainer()      
        self.clf = tr.load_model(model_path)
        
    def clf_quality(self,X_test,y_test):
        """评估分类器效果"""
        pred = self.clf.predict(X_test)
        cnf_matrix = confusion_matrix(y_test, pred)
        score = self.clf.score(X_test, y_test)
        clf_repo = classification_report(y_test, pred)
        return cnf_matrix,score,clf_repo
    
    def predict(self, fn):
        '''样本预测;'''
        pt = Preprocessor()
        tmp = pt.img2vec(fn)
        X_test = tmp.reshape(1, -1)
        ans = self.clf.predict(X_test)
        return ans

def run_train():
    t0 = time.time()
    pt = Preprocessor()
    tr = Trainer()
    
    X_train, y_train = pt.get_data_labels()
    X_test, y_test = pt.get_data_labels("test")
    
    t1 = time.time()
    print(t1 - t0)
    clf = tr.svc(X_train, y_train)
    print(time.time() - t1)

    tr.save_model(clf, "mnist_svm.m")
    
    tester = Tester("mnist_svm.m")
    mt, score, repo = tester.clf_quality(X_test, y_test)
    print(mt, score, repo)
    return clf
    