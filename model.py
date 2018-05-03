#-*- coding: utf-8 -*-

import numpy as np
import os
from PIL import Image
import random
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.externals import joblib

file_path = r'dataset\mnist_data'

def img2vec(fname):
    '''将png等格式的图片转为向量'''
    im = Image.open(fname).convert('L')
    im = im.resize((28,28))
    tmp = np.array(im)
    vec = tmp.ravel()
    return vec

def split_data(paths):
    '''随机抽取1000张图片作为训练集'''
    fn_list = os.llistdir(paths)
    X = []
    y = []
    d0 = random.sample(fn_list,1000)
    for i,name in enumerate(d0):
        y.append(name[0])
        X.append(img2vec(name))
        dataset = np.array([X,y])
    return X,y

def knn_clf(X_train,label):
    '''构建分类器'''
    clf = knn()
    clf.fit(X_train,label)
    return clf

def save_model(model,output_name):
    '''保存模型'''
    joblib.dump(model,ouotput_name)

def load_model(model_path):
    '''加载模型'''
    clf = joblib.load(model_path)
    return clf

def tester(fn,clf):
    '''预测数据'''
    tmp = img2vec(fn)
    X_test = tmp.reshape(1,-1)
    ans = clf.predict(X_test)
    return ans

def get_data():
    '''用sklearn提供的函数下载mnist数据集'''
    from sklearn.datasets import fetch_mldata
    from sklearn.cross_validation import train_test_split
    data = fetch_mldata('MNIST original', data_home='dataset\mnist_data')
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.65)
    return X_train,y_train

##X_train,y_label = get_data()

##X_train,y_label = split_data(file_path)
##clf = knn_clf(X_train,y_label)
##save_model(clf,'mnist_knn1000.m')
