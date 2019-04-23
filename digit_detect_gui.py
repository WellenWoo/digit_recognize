#-*- coding: utf-8 -*-

import wx
from collections import namedtuple
from PIL import Image
import zbar
import os

import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.externals import joblib
import wx.lib.agw.gradientbutton as gbtn
import model
"""
recognize the digits basic on sklearn
"""
__author__ = 'wellenwoo'
__mail__ = 'wellenwoo@163.com'

origin_path = os.getcwd()
wildcard ="jpg (*.jpg)|*.jpg|" \
           "png(*.png) |*.png|"\
           "jpeg(*.jpeg) |*.jpeg|"\
           "tiff(*.tif) |*.tiff|"\
           "All files (*.*)|*.*"

class MainWindow(wx.Frame):
    def __init__(self,parent,title):
        wx.Frame.__init__(self,parent,title=title,size=(600,-1))
        static_font = wx.Font(12, wx.SWISS, wx.NORMAL, wx.NORMAL)
        
        Size = namedtuple("Size",['x','y'])
        s = Size(100,50)

        self.fileName = None
        self.model = model
        
        b_labels = [u'open',u'run']

        TipString = [u'选择图片', u'识别数字']
        
        funcs = [self.choose_file,self.run]
        
        '''create input area'''
        self.in1 = wx.TextCtrl(self,-1,size = (2*s.x,3*s.y))
        self.out1 = wx.TextCtrl(self,-1,size = (s.x,3*s.y))

        '''create button'''
        self.sizer0 = wx.FlexGridSizer(cols=4, hgap=4, vgap=2)
        self.sizer0.Add(self.in1)
        
        buttons = []
        for i,label in enumerate(b_labels):
            b = gbtn.GradientButton(self, id = i,label = label,size = (1.5*s.x,s.y))
            buttons.append(b)
            self.sizer0.Add(b)      

        self.sizer0.Add(self.out1)

        '''set the color and size of labels and buttons'''  
        for i,button in enumerate(buttons):
            button.SetForegroundColour('red')
            button.SetFont(static_font)
            button.SetToolTipString(TipString[i])
            button.Bind(wx.EVT_BUTTON,funcs[i])

        '''layout'''
        self.SetSizer(self.sizer0)
        self.SetAutoLayout(1)
        self.sizer0.Fit(self)
        
        self.CreateStatusBar()
        self.Show(True)
    
    def run(self,evt):
        if self.fileName is None:
            self.raise_msg(u'请选择一幅图片')
            return None
        else:
            data = self.parse_QR(self.fileName)
            if len(data)>0:
                self.out1.Clear()
                self.out1.write(data)
            else:
                model_path = os.path.join(origin_path,'mnist_knn1000.m')
                clf = model.load_model(model_path)
                ans = model.tester(self.fileName,clf)
                self.out1.Clear()
                self.out1.write(str(ans))
        
    def choose_file(self,evt):
        '''choose img'''
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=os.getcwd(), 
            defaultFile="",
            wildcard=wildcard,
#            style=wx.OPEN | wx.MULTIPLE | wx.CHANGE_DIR #wx2.8
            style = wx.FD_OPEN | wx.FD_MULTIPLE |     #wx4.0
                    wx.FD_CHANGE_DIR | wx.FD_FILE_MUST_EXIST |
                    wx.FD_PREVIEW
            )
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            dlg.Destroy()
            self.in1.Clear()
            self.in1.write(paths[0])
            self.fileName = paths[0]
            im = Image.open(self.fileName)
            im.show()
        else:
            return None
    
    def parse_QR(self,fname):
        scanner = zbar.ImageScanner()
        scanner.parse_config("enable")
        pil = Image.open(fname).convert('L')
        width, height = pil.size
        raw = pil.tostring()
        image = zbar.Image(width, height, 'Y800', raw)
        scanner.scan(image)
        data = ''
        for symbol in image:
            data+=symbol.data
        del(image)
        return data
    
    def raise_msg(self,msg):
        '''warning message'''
        info = wx.AboutDialogInfo()
        info.Name = "Warning Message"
        info.Copyright = msg
        wx.AboutBox(info)
        
if __name__ == '__main__':
    app = wx.App(False)
    frame = MainWindow(None,'Digit Recognize')
    app.MainLoop()
