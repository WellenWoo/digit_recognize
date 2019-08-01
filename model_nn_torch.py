# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:05:03 2019
@author: WellenWoo
"""
import torch
from torch.autograd import Variable
import numpy as np
import time
from utils import Preprocessor, make_shuffle

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        keep_prob = 0.7
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 28, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p = 1-keep_prob))
        
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(28, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p = 1-keep_prob))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2,padding = 1),
            torch.nn.Dropout(p = 1-keep_prob))
        
        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(in_features = 4 * 4 * 128, out_features= 625, bias = True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
                self.fc1,
                torch.nn.ReLU(),
                torch.nn.Dropout(p = 1-keep_prob))
         
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(in_features=625, out_features=10,bias = True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out
        
class Trainer(object):
    """训练器"""
    def net(self,X_train,y_train,lr = 1e-3,epochs = 10, device = "cuda:0"):
        """y_train无需one hot encode"""
        if not isinstance(X_train,torch.Tensor):
            X_train = Variable(torch.Tensor(X_train)).to(device)
            y_train = Variable(torch.Tensor(y_train)).to(device)
        else:
            X_train = X_train.to(device)
            y_train = y_train.to(device)
        batch_size = 64

        model = CNN().to(device)
        
        criterion = torch.nn.CrossEntropyLoss()
        optm = torch.optim.Adam(model.parameters(), lr = lr)
        
        for step in range(epochs):
            avg_cost = 0
            total_batch = len(X_train)//batch_size
            
            for i in range(total_batch):
                start = i*batch_size
                end = (i+1)*batch_size
                
                x = X_train[start:end]
                y = y_train[start:end]
                optm.zero_grad()
                hypothesis = model(x)
                cost = criterion(hypothesis, y.long().view(-1))
                cost.backward()
                optm.step()
                
                avg_cost += cost.data / total_batch
                
            print("[Epoch: {:>4}] cost = {:>.9}".format(step + 1, avg_cost.item()))                
        return model 

class Tester(object):
    """测试器"""
    def get_acc(self, clf, X_test, y_test, device = "cuda:0"):
        if not isinstance(X_test,torch.Tensor):
            X_test = Variable(torch.Tensor(X_test)).to(device)
            y_test = Variable(torch.Tensor(y_test)).to(device)
        else:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
        clf.eval()    
        with torch.no_grad():
            pred = clf(X_test)
            pred_inverse_one_hot = torch.max(pred.data,1)[1].float()
            correct_pred = (pred_inverse_one_hot == y_test.data)
            acc = correct_pred.float().mean()
        return acc 
    
def run():
    pt = Preprocessor()    
    tr = Trainer()
    ts = Tester()  
    t0 = time.time()
    X_train, y_train = pt.load_data()
    X_test, y_test = pt.load_data("mnist_test_data.npz")
    
    X_train, y_train = make_shuffle(X_train, y_train)
    X_test, y_test = make_shuffle(X_test, y_test)
    
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))
    print(time.time() - t0)
    t1 = time.time()
    clf = tr.net(X_train, y_train)
    print(time.time() - t1)
    acc = ts.get_acc(clf, X_test, y_test) #acc=97.8%
    
    return clf, acc