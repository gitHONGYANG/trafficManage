from core.service import ImageReceiver
import time
import datetime
import uuid  #生成唯一标识符
from core.config import rec_img_tb, log_db
import cv2
import os
from core.config import _recimgsavepath
from PIL import Image
import base64   #使用base64编码解码
from urllib import request, parse
import numpy as np
from utils.upload_img import upload
import threading

class SaveRecentImg(ImageReceiver):    # from core.service import ImageReceiver  继承于ImageReceiver
    def __init__(self, ID, cameraid):    #获得参数：摄像头序号，摄像头号
        #这里竟然没有初始化父类
        self.id = ID
        self.info = None
        self.savepath = _recimgsavepath +cameraid +'/'    #from core.config   源地址：_recimgsavepath = '/media/assests/Recimgs/' + datestr + '/'     datestr是获得的时间
        self.cameraid = cameraid
        self.tb = log_db[cameraid + '_rec']   #from core.config import rec_img_tb, log_db
        # 原定义：log_db = MongoClient('localhost:27017')['log']   从数据库里处理数据 作为日志

        self.lastt = 0   #上次操作的时间
        self.reccount = 0        


    def feed(self, info):  #问题：info的类型？？
        #feed函数的作用  ，将图像上传并保存到数据库
        frame = info['img']
        imageid = str(uuid.uuid1())    #生成一个唯一的id号
        
        t = info['time']       #得到时间
        cameraid = self.cameraid
        imagepath = self.savepath + self.cameraid + '_' + str(t) + '.jpg'
        
        lowppi_frame = cv2.resize(frame, (800, 450))    #图像重构
        ret, buffer = cv2.imencode('.jpg', lowppi_frame)  #cv2.imencode()函数是将图片格式转换(编码)成流数据，
        # 赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输，ret表示转换成功
        if ret:
            b64img = base64.b64encode(buffer)   #编码
            self.tb.insert_one({'time':t, 'lastt':self.lastt,'dt':t-self.lastt,
                                'content':'saveimg','imagepath':imagepath})  #更新新操作日志
            
            up_thread = threading.Thread(target=upload, args=(imagepath, b64img))   #创建线程
            # from utils.upload_img import upload  上传图片，  上传图片是典型的IO操作，使用多线程加快效率
            up_thread.start()
            #upload(imagepath, b64img)
            recent_img_info = {
                'imageid':imageid,
                'time':t,
                'cameraid':cameraid,
                'imagepath':imagepath,
                'info':''
                }
            rec_img_tb.insert_one(recent_img_info)  #rec_img_tb = MongoClient(url)[dbname][rectb_name]将图像信息插入数据库
            self.reccount += 1
            if self.reccount%(60) == 0:
                print(self.cameraid + '  recimg\t' + str(self.reccount))
            self.lastt = t    #跟新上次操作的时间
