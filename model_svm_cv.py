# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:48:45 2019

@author: Administrator
"""
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report
import time
from utils import Preprocessor
import cv2
from model_svm import Tester
import sklearn

class Trainer(object):
    '''训练器;'''
    def svc(self,X_train,label,kernel = cv2.ml.SVM_POLY,
            Type = cv2.ml.SVM_C_SVC, C = 1.0, gamma = 1.0,
            degree = 3, coef0 = 0):
        """cv的SVM分类器"""
        X_train = X_train.astype('float32')
        label = label.reshape(-1,1)
        clf = cv2.ml.SVM_create()
        clf.setKernel(kernelType =kernel )
        clf.setType(Type)
        clf.setC(C)
        clf.setGamma(gamma)
        clf.setDegree(degree)
        clf.setCoef0(coef0)
        clf.train(X_train,cv2.ml.ROW_SAMPLE,label)
        return clf 
    
    def save_model(self,model,output_name):
        '''保存模型'''
        if isinstance(model,sklearn.svm.classes.SVC):
            joblib.dump(model,output_name,compress = 1)
        elif isinstance(model,cv2.ml_SVM):
            model.save(output_name)

    def load_model(self,model_path):
        '''加载模型'''
        try:
            clf = joblib.load(model_path)
        except Exception:
            clf = cv2.ml.SVM_load(model_path)
        return clf   

class Tester_cv(Tester):
    '''测试器;'''
    def __init__(self, model_path):
        tr = Trainer()      
        self.clf = tr.load_model(model_path)
        
    def clf_quality(self,X_test,y_test):
        """评估分类器效果"""
        pred = self.clf.predict(X_test)
        cnf_matrix = confusion_matrix(y_test, pred)
        
        clf_repo = classification_report(y_test, pred)
        return cnf_matrix, -1, clf_repo
    
    def predict(self, fn):
        '''样本预测;'''
        pt = Preprocessor()
        tmp = pt.img2vec(fn)
        X_test = tmp.reshape(1, -1)
        X_test = X_test.astype("float32")
        ans = self.clf.predict(X_test)
        return ans