# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:59:54 2019
@author: WellenWoo
以mnist数据集中数字8,9两个类别各20张图像作为
source domain 训练模型,
利用该模型进行迁移学习,用以分类mnist数据集中数字0~7,
target domain 中共8个类别,每个类别各20个样本,
val_acc 可达 94.5%.
"""
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, models, transforms
import time
from tool import files2sample
from PIL import ImageFile
import torch
from PIL import Image
from model_nn_torch_DataLoader import CNN
from utils import file2tensor, load_img

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=30, decay_weight = 0.1):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (decay_weight**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

class Net(object):
    def __init__(self, network,state_dict_path = r"cnn_mnist_89sample45epoch24acc84.pth", 
                 device = "cuda:0"):
        
        self.device = device
        self.model = network
        self.model.load_state_dict(torch.load(state_dict_path))
    
    def transfer(self, num_classes):        
        fc_in = self.model.fc2.in_features
        self.model.fc2 = nn.Linear(fc_in, num_classes).to(self.device)
        
    def fit(self, data_loader, epochs = 100):
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        num_classes = len(data_loader.dataset.classes)
        self.transfer(num_classes)
                
        optimizer = optim.RMSprop(self.model.parameters(), lr=0.0001)
        
        for epoch in range(epochs):
            optimizer = exp_lr_scheduler(optimizer, epoch)
            self.model.train()
            
            running_loss = 0.0
            running_acc = 0
            counter = 0
            
            for data, label in data_loader:
                data = Variable(data.to(self.device))
                label = Variable(label.to(self.device))
                
                optimizer.zero_grad()
                out = self.model(data)
                _, pred = torch.max(out.data, 1)
                
                loss = criterion(out, label)
                
                counter += 1
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                running_acc += torch.sum(pred == label.data)
                
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_acc.item() / float(len(data_loader.dataset))
            print("Loss:{:.4f} Acc:{:.4f}".format(epoch_loss, epoch_acc))
                
    def evaluate(self, data_loader):        
        self.model.eval()
        acc = 0
        for data, label in data_loader:
            data = Variable(data.to(self.device))
            label = Variable(label.to(self.device))
            
            with torch.no_grad():
                out = self.model(data)
                _, pred = torch.max(out.data, 1)
                correct_pred = (pred == label.data)
                acc += torch.sum(correct_pred)
        acc = acc.float()/len(data_loader.dataset.samples)
        return acc
    
    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            _, pred = torch.max(out.data, 1)
        return pred

def run():
    train_source_model()
    train_loader = load_img(r"mini_train_07")
    val_loader = load_img(r"test")
    
    clf = Net(CNN(), r"cnn_mnist_89sample45epoch24acc84.pth")
    clf.fit(train_loader)
    acc = clf.evaluate(val_loader)
    
    return clf, acc

def train_source_model():
    from model_nn_torch_DataLoader import Trainer
    tr = Trainer()
    
    train_loader = load_img(r"mini_train_89")
    clf = tr.net(train_loader)
    torch.save(clf.state_dict(), r"cnn_mnist_89sample45epoch24acc84.pth")
    