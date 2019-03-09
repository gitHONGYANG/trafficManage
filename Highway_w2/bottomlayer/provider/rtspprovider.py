from core.service import ImageProvider
import cv2
import time
import pymongo
from pymongo import MongoClient    #MongoDB库
import base64
from PIL import Image
from io import BytesIO
import numpy

class RtspProvider(ImageProvider):
    def __init__(self, id, cameraid):
        self.id = id  #确定服务器的编号
        self.imgtb = MongoClient('localhost:27017')['highway']['temp_img']   #连接
        self.cameraid = cameraid

    def impulse(self):
        #将img，t格式化输出
        img, t = self.get_newimg()
        return {'id': self.id, 'img': img, 'time': t}

    def get_newimg(self):
        #从数据库中得到图片
        info = self.imgtb.find_one({'cameraid' : self.cameraid},{'tempimg' : 1, 'time':1})  #得到图片
        img = info['tempimg']
        t = info['time']
        img = Image.open(BytesIO(base64.b64decode(img)))  #因为图片是以base64的编码存储的，所以在提取的时候解码
        img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)   #将RGB转成BGR格式
        return img, t

    def ok(self):
        self.status = True
        #self.status, self.frame = self.cap.read()
        if not self.status: print('video end')
        return self.status
