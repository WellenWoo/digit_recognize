# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:17:38 2019

@author: WellenWoo
"""
import numpy as np
import os
from PIL import Image
from glob import glob
import torch
import torchvision as tv

def make_shuffle(x, y):
    """对数据进行洗牌;
    args:
    ------
        x: 图片数据,(batch, weight * height),
        y: 图片便签,(batch),
    
    return:
        x,y:洗牌后的数据;
    
    example:
        x = np.arange(28).reshape(7,4)
        y = np.arange(50, 57)
        x, y = make_shuffle(x, y);"""
    data = np.c_[x, y]
    np.random.shuffle(data)
    
    nh = range(data.shape[0])
    nv = range(data.shape[1] - 1)
    
    x = data[nh][:,[nv]]
    x = x.squeeze(1)
    
    y = data[nh][:,-1]
    
    return x, y

def load_img(fpath, batch_size = 64, workers = 4):
    """将文件夹中不同子文件夹的图片转为数据集;
    传入的参数为r'mnist';
    子文件夹分别为r'mnist/0',r'mnist/1,...';
    """
    trans = tv.transforms.Compose([
            tv.transforms.Grayscale(),
            tv.transforms.ToTensor()])
    
    dataset = tv.datasets.ImageFolder(fpath, transform=trans)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers= workers,
                                             drop_last = False)
    
    return dataloader

class Preprocessor(object):
    """训练前的预处理"""
    def get_files(self,fpath,fmt = "*.png"):
        """获取指定文件夹中指定格式的文件列表;
        paras:
            filepath: str, file path,
            formats: str, file format,
        return: list;"""
        tmp = os.path.join(fpath,fmt)
        fs = glob(tmp)
        return fs
    
    def get_data_labels(self, fpath = "train"):
        paths = glob.glob(fpath + os.sep + "*")
        X = []
        y = []
        for fpath in paths:
            fs = self.get_files(fpath)
            for fn in fs:
                X.append(self.img2vec(fn))
            label = np.repeat(int(os.path.basename(fpath)), len(fs))
            y.append(label)
        labels = y[0]
        for i in range(len(y) - 1):
            labels = np.append(labels, y[i + 1])
        return np.array(X), labels
    
    def img2vec(self, fn):
        '''将jpg等格式的图片转为向量'''
        im = Image.open(fn).convert('L')
        im = im.resize((28,28))
        vec = np.array(im)
        vec = vec.ravel()
        return vec 
    
    def save_data(self, X_data, y_data, fn = "mnist_train_data"):
        """将数据保存到本地;"""
        np.savez_compressed(fn, X = X_data, y = y_data)
        
    def load_data(self, fn = "mnist_train_data.npz"):
        """从本地加载数据;"""
        data = np.load(fn)
        X_data = data["X"]
        y_data = data["y"]
        return X_data, y_data