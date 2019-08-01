# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:05:03 2019
@author: WellenWoo
"""
import torch
from torch.autograd import Variable
import numpy as np
import torchvision as tv
import torch.nn.functional as F
import torchsnooper
from utils import load_img

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
    def net(self, data_loader, lr = 1e-3, epochs = 10, device = "cuda:0"):
        """y_train无需one hot encode"""  
        model = CNN().to(device)
        
        criterion = torch.nn.CrossEntropyLoss()
        optm = torch.optim.Adam(model.parameters(), lr = lr)
        
        for step in range(epochs):
            avg_cost = 0
            total_batch = len(data_loader)
            
            for i, (batch_xs, batch_ys) in enumerate(data_loader):
                x = Variable(batch_xs.to(device))
                y = Variable(batch_ys.to(device))                
                
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
#    @torchsnooper.snoop()
    def get_acc(self, clf, data_loader, device = "cuda:0"): 
        clf.eval()   
        acc = 0
        for data, label in data_loader:
            data, label = data.to(device), label.to(device)
            with torch.no_grad():
                pred = clf(data)
                pred_inverse_one_hot = torch.max(pred.data,1)[1].long()
                correct_pred = (pred_inverse_one_hot == label.data)
                acc += torch.sum(correct_pred)
        acc = acc.float()/len(data_loader.dataset.samples)
        return acc
    
def run():
    tr = Trainer()
    ts = Tester()  
    
    train_loader = load_img(r"D:\..MNIST_data\train")
    test_loader = load_img(r"D:\..MNIST_data\test")  
    
    clf = tr.net(train_loader)
    acc = ts.get_acc(clf, test_loader) #acc=99.39%
    
    return clf, acc