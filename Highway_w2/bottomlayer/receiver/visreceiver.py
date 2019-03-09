#本脚本的作用是进行能见度检测

from core.service import ImageReceiver
from core.config import vis_tb, vismodelpath, log_db
import time
import  torch    #用到了torch0.3版本
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import os
from PIL import Image    #获得图像
import math
import datetime
import cv2
import random

class Net (nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained =True)   #使用网络内置的resnet18网络
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 16)    #替换掉源代码的linear层
        self.FC1 = nn.Linear(16, 1)    #作为回归器

    def forward(self, x):
        output = self.model(x)
        output = F.sigmoid(output)
        output = self.FC1(output)
        return output

class getvisibility():    #得到可见度，注意，这个函数处理的是图像
    def __init__(self, modelpath):
        self.model = Net()
        self.model.load_state_dict(torch.load(modelpath))    #导入预先训练好的参数
        self.model.cuda() #gpu版本
        self.transform = transforms.Compose(       #预处理，重构大小，标准化
            [transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    def getfromcv(self, cvimage):
        pilimage = Image.fromarray(cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB))      #将接受的image转换维RGB版本，并将其转换成数组

        image = self.transform(pilimage)    #预处理
        image = image.unsqueeze(0)      #增加维度，没有batch，直接处理一张图
        image = Variable(image).cuda()
        output = self.model(image).cpu()   #获得结果

        vis = float(output[0][0]) * 1000    #这里tensor转换？？？？


        #这个上下限的处理是干啥？？？
        if vis<200:
            vis = 200 + random.randint(-100, 100)
        elif vis>10000:
            vis = 10000 + random.randint(-1000, 1000)

        return vis

class SaveVis(ImageReceiver):       #存储能见度信息，并将值写入数据库，更新系统日志
    def __init__(self, id, cameraid):
        self.info = None
        self.cameraid = cameraid
        self.id = id
        self.logtb = log_db[cameraid+'_vis']
        self.lastt = 0        
        self.viscount = 0                   #处理的次数
        if not hasattr(SaveVis, 'process'):
            SaveVis.process = getvisibility(vismodelpath)   #类定义
        self.visibility = 0
        self.lastvisibility = 0        #我认为这个变量没什么意义

    def feed(self, info):    #对数据库的操作
        curt = info['time']
        if self.viscount == 0:        #还未进行检测
            self.visibility = SaveVis.process.getfromcv(info['img'])  #得到可见度
            self.lastvisibility = self.visibility
        else:
            vis = SaveVis.process.getfromcv(info['img'])        #检测可见度
            self.visibility = self.lastvisibility*0.7 + vis*0.3     #新的可见度由旧的可见度与新的可见度组合二场
            self.lastvisibility = self.visibility
        
        rank = 3                              #根据能见度对当时的天气进行了分级
        if self.visibility > 2600:
            rank = 3
        elif self.visibility>1400:
            rank = 2
        elif self.visibility>800:
            rank = 1
        else:
            rank = 0

        vis_info = {'cameraid':self.cameraid,
                'visibility':self.visibility,
                'time':curt,
                'saferank':rank,
                'info':''
                }
        vis_tb.insert_one(vis_info)    #将能见度写入数据库
        self.logtb.insert_one({'time':curt, 'dt':curt-self.lastt, 'content':'upload visibility','visibility':self.visibility}) #更新日志
        self.viscount += 1
        if self.viscount%20 == 0:     #每20次操作输出一次
            print(self.cameraid + '  vis\t', str(self.viscount))
        self.lastt = curt

