# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:19:34 2019

@author: hongyang
"""

#本程序相对于以往加入数据库
#文件夹目录的写入，读取，查重
#图片目录的读入与查重
#使用数据库trafficManager，集合按摄像头分类
import pymongo
import cv2
import os
import numpy as np
import re
from threading import Thread
from time import sleep
#vediodir = 'vedio/'  # 存放vedio的目录文件夹 

class vediotopics:
    #返回数据库的型号，得到src_path
    def __init__(self, c_vediodir=''):
        self.vediodir = c_vediodir
        self.vedios = []  # 保存vedio
        self.vedio_path = []
        

    def getvediopath(self):   
        #得到各个MP4文件的所在位置
        for root, dirs, files in os.walk(top=self.vediodir):
            for file in files:
                if os.path.splitext(file)[1] == '.mp4':  # 限制文件类型
                    self.vedios.append(os.path.join(root, file))
                    filepath = os.path.join(root, file)
                    print(filepath)   #得到完整文件名
                    self.vedio_path.append(filepath)
        return self.vedio_path
    def getvedio(self):
        #调用getvedio时必须先调用getvediopath
        vedio_roi = []   #过渡数组
        if len(self.vedio_path) == 0:
            raise ValueError("there is no information about vedio")
        else:
            for vpath in self.vedio_path:
                print(vpath.split('\\')[-2])
                vedio_roi.append(vpath.split('\\')[-2])
            vedioset = set(vedio_roi)   #去除重复值
            self.vedios = list(vedioset)
            del vedioset
            del vedio_roi    #将无用变量去掉
            self.vedios.sort()     #对摄像头号进行排序
        return self.vedios


                    
class Imgsplit:      #一个vedio得到一个类
    def __init__(self,vedio,vediopath,db=None,object_dir=''):  #vedio是一个str
        self.object_dir = object_dir   #目的文件夹
        self.vedioname = vedio
        self.dbtable = db[vedio]       #数据库导入,形成相应的集合（这里沿用老名称table）
        self.vediopath = self.selectpath(vediopath)
    def selectpath(self,vediopath):   #对vediopath进行筛选
        pathes = []
        for path in vediopath:
            try:
                re.search(self.vedioname,path).span()
                pathes.append(path)   #找到了，加入列表
            except AttributeError:  #若未找到匹配字符会触发异常
                print("don't math ",self.vedioname,'in ',path)
                pass                 #没找到，不加
        return pathes
    def getimg_and_save(self):
       # img = []  #存储生成的imgge
        for vedio_p in self.vediopath:
            vedio = cv2.VideoCapture(vedio_p)
            ret, frame = vedio.read()
            count = 0
            frequency = 0   #截图的频率
            max_frequency = 3000   #每三千帧截一次图
            while ret and not keyerror:
              #  vedio.set(cv2.CAP_PROP_POS_MSEC,500 * 1000 * count)  #弃用，因为读取的文件有坏帧，可能出现bug
                savepath = self.object_dir + self.vedioname + '\\' + ((os.path.split(vedio_p)[1]).split('.')[0]).split('\\')[-1] + '\\'
                
                if not os.path.exists(savepath):   #判断是否存在此文件夹
                    os.makedirs(savepath)    #不存在，则创建一个
                    print(savepath)

                frequency += 1
                if frequency % 1000 == 0: print(frequency)
                if frequency == max_frequency:
                    
                    filepath = savepath + '{}.jpg'.format(count)
                    cv2.imencode('.jpg', frame)[1].tofile(filepath)
               #     cv2.imwrite(filepath, frame)  # 注意cv2不会自主创建目录   imwrite对中文路径不支持，弃用
                    insertone = {
                            "imagepath":filepath,
                            "camera":self.vedioname
                    }
                    #查重
                    self.dbtable.insert_one(insertone)
                    frequency = 0
                    print(self.vedioname + "the %dth picture" % count)
                    count += 1
                    
                ret, frame = vedio.read()

        


def main():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")  #连接数据库
    print("link database")
    mydb = myclient["trafficManager"]   #调用相关数据库
    v = vediotopics(c_vediodir='vedio')
    mvediopath = v.getvediopath()
    vedios = v.getvedio()
    imgspli = Imgsplit(vedio=vedios[0],vediopath=mvediopath,db=mydb,object_dir='img\\')
#    imgspli.getimg_and_save()
   
       
    threads = []
    for mvedio in vedios:
        
        imgspli = Imgsplit(vedio=mvedio,vediopath=mvediopath,db=mydb,object_dir='img\\')
        thread = Thread(target = imgspli.getimg_and_save,args=())
        threads.append(thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

       
    

    
if __name__ == '__main__':
    main()

