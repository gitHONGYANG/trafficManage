#carmatchreceiver.py 作用：车辆位置的截取，特征的抽取

from core.service import ImageReceiver   #基类
from core.config import car_tb, tvs_alwayson
from car_match.yolo2py.darknet import Darknet19   #得到模型
import car_match.yolo2py.utils.yolo as yolo_utils
import car_match.yolo2py.utils.network as net_utils
import car_match.yolo2py.cfgs.config as cfg
from PIL import Image    #处理图像
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable
from core.config import carinfo_tb, similaritymodelpath, yolomodelpath, log_db, geo_loc
import os
import uuid    #生成唯一的标识符
import datetime
import time
import base64   #转码
from utils.upload_img import upload   #上传图片
import threading

class yoloNet():
    def __init__(self, modelpath):
        self.thresh = 0.5
        self.label = ('car', 'truck', 'trailer', 'bus')   #四种类型车辆
        self.net = Darknet19()
        self.net.load_state_dict(torch.load(yolomodelpath))  #导入模型数据
        #net_utils.load_net(modelpath, self.net)
        self.net.cuda()
        self.net.eval()  #模型固定
        print('Yolo init Success')

    def preprocess(self, content, method = 'nparray'):   #数据预处理
        image = 0
        if method == 'path':      #根据传入的参数不同对image进行不同的处理
            image = cv2.imread(content)
        elif method == 'nparray':
            image = content
        else:
            print('\n\nError in preprocess\n\n')     #参数错误   这里考虑用异常处理是不是要好点？？
        #yolo_utils.preprocess_test((image, none, cfg.multi_scale_inp_size), 0)将图像转换到RGB空间,resize到(320, 320),值归一化到(0,1)返回的0号元素为处理后图像,4号元素为原图拷贝
        im_data = np.expand_dims(yolo_utils.preprocess_test((image, None, cfg.multi_scale_inp_size), 0)[0], 0)  #拓展维度
        #im_data被转换为(1,3,320,320)归一化np矩阵 yolo_utils.preprocess_test？？
        return image, im_data

    #输入path
    def getCarinfofromPic(self, content, method = 'nparray'):     #从图像中得到车的信息
        image, im_data= self.preprocess(content, method = method)
        im_data = net_utils.np_to_variable(im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
        #进一步处理，permute的作用 将pytorch的数组的维度换位
        bbox_pred, iou_pred, prob_pred = self.net(im_data)     #可以了解到bbox输出的是一个矩形坐标信息，代表找到的车辆位置，
        # iou_prod和prob_pred分别代表该预测的位置(bbox)的置信度与预测的物体类型置信度(yolo可以同时预测多种物体位置)。

        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()    #转换成numpy

        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, image.shape, cfg, self.thresh,
                                                          size_index=0)   #得到位置与相应的种类
        #postprocess？？？

        roi = []    #roi干啥的？？形成数据流？？
        for i in range(len(bboxes)):     #bbox存储的是车在照片的像素位置
            roiimage = image[bboxes[i][1]:bboxes[i][3], bboxes[i][0]:bboxes[i][2]]   #将车的部分截取出来
            roi.append(roiimage)

        return bboxes, scores, cls_inds, image, roi

    #不返回car
    def getCarROIpil(self, content, method = 'nparray'):
        bboxes, scores, cls_inds, ori_image, roi = self.getCarinfofromPic(content , method)
        newbboxes , newscores, newcls_inds, newori_image, newroi = [], [], [], [], []   #将原数据转换成数列
        carImage = []

        cars_loc = []
        for i in range(len(cls_inds)):     #cls_ids 的长度和roi的长度应该是一致的
            if cls_inds[i]!=0:
                pilroi = Image.fromarray(cv2.cvtColor(roi[i],cv2.COLOR_BGR2RGB))  #照片色度处理
                carImage.append(pilroi)
                newbboxes.append(bboxes[i])
                newscores.append(scores[i])
                newcls_inds.append(cls_inds[i])
            cars_loc.append(bboxes[i])
        return carImage, newbboxes, newscores, newcls_inds, cars_loc

class BaseNetwork(nn.Module):    #比较相似性

    def __init__(self, modelname):
        super(BaseNetwork, self).__init__()
        self.modelname = modelname
        if modelname == 'Vgg16':
            self.CNN = models.vgg16(pretrained=True).features
            self.FC1 = nn.Linear(7*7*512, 2048)
            self.FC2 = nn.Linear(2048, 128)
        else:

            raise ('Please select model')

    def forward(self, x):
        if self.modelname == 'Vgg16':
            output = self.CNN(x)
            output = output.view(output.size()[0], -1)
            output = self.FC1(output)
            output = F.relu(output)
            output = self.FC2(output)
        else:
            raise ('Please select model')
        return output



class similarity():   #得到两个东西的相似性,以找到匹配车辆
    def __init__(self, modelpath):
        self.model = BaseNetwork('Vgg16')  #使用成熟的vgg16网络
        self.model.load_state_dict(torch.load(modelpath))
        self.model.cuda()
        #self.tranform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.ColorJitter(), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.tranform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def getEmbeddingformPILlist(self, pillist):
        featurelist = []

        for pilimage in pillist:
            tensorimage =  Variable(self.tranform(pilimage)).view(1, 3, 224, 224).cuda()  #batch =1 ，inchannel = 3
            feature = self.model(tensorimage)    #获得特征
            featurelist.append(feature)
        return featurelist

class SaveCarInfo(ImageReceiver):          #主类
    def __init__(self, ID, savepicpath, cameraID):
        self.info = None
        self.cartb = log_db[cameraID + '_car']
        if not hasattr(SaveCarInfo, "yolo"):
            SaveCarInfo.yolo = yoloNet(yolomodelpath)    #yolo网络
        if not hasattr(SaveCarInfo, "simi"):             #simi网络
            SaveCarInfo.simi = similarity(similaritymodelpath)

        self.savepicpath = savepicpath
        self.cameraid = cameraID
        self.id = ID
        self.lastt = 0
        self.carcount = 0
        self.ifsavefeature = True if (self.cameraid in tvs_alwayson) else False   #tvs_alwayson 原定义： tvs_alwayson = ['TV%d' %i for i in range(52, 72)]

    def feed(self, info):
        frame = info['img']

        carImagelist, bboxes, scores, cls_inds, cars_loc = SaveCarInfo.yolo.getCarROIpil(frame, 'nparray')  #得到carimage，车的位置
        
        ifsaveimage = False
        picpath = self.savepicpath
        imageid = str(uuid.uuid1())   #得到唯一的id值
        curt = info['time']
        imagepath = picpath + '/' + self.cameraid  + '_'  + str(curt) +'.jpg'


        #保存cars_loc
        if len(cars_loc)>0:
            ifsaveimage = True
            print(imagepath)
            self.cartb.insert_one({'time':curt, 'dt':curt-self.lastt,'content':'upload car', 'imagepath':imagepath})   #将操作写入日志
        
        for car_loc in cars_loc:
            cameraid = self.cameraid
            location = car_loc
            location = [int(l) for l in location]
            info = {'cameraid':cameraid,
                    'imagepath':imagepath,
                    'time':curt,
                    'location':location,
                    'info':''
                    }
            car_tb.insert(info)   #将车的位置时间信息存进数据库
                        

        
        #保存feature信息
        if self.ifsavefeature:
            featurelist = SaveCarInfo.simi.getEmbeddingformPILlist(carImagelist)  #得到特征
            carinfolist = []
            for i, feature in enumerate(featurelist):

                carid = str(uuid.uuid1())     #得到唯一车标识
                # feature 转化为一个128维[float, float],供数据库存储
                feature = list(feature.cpu().data.numpy().flatten())
                feature = [int(f*1000) for f in feature]   #对特称做一下处理
                location = bboxes[i]
                location = [int(l) for l in location]           #感觉这里很繁琐
                cameraid = self.cameraid
                area = (location[2]-location[0])*(location[3]-location[1])

                y = (location[1] + location[3])/2
                y = int((1080-y)/1080 * 100 + 20)  #标准化
                pos_1, pos_2 = geo_loc[cameraid]  #geo_loc 各个摄像头的信息
                pos_2 += y
                if pos_2>1000:      #超过了探测距离
                    pos_2 -= 1000
                    pos_1 += 1
                pos = 'K' + str(pos_1) + '+' + str(pos_2)
            
                if area>150*150:
                    carinfo = {'carid' : carid,
                            'imageid' : imageid,
                            'time' : curt,
                            'feature': feature,
                            'cameraid': cameraid,
                            'location' : location,
                            'position' : pos,
                            'info' : '',
                            'imagepath' : imagepath
                            }
                    carinfolist.append(carinfo)

        if ifsaveimage:
            print('upload image')
            
            for carinfo in carinfolist:
                carinfo_tb.insert_one(carinfo)   #数据库输入
            lowppi_frame = cv2.resize(frame, (800, 450))
            ret, buffer = cv2.imencode('.jpg', lowppi_frame)
            if ret:
                b64img = base64.b64encode(buffer)
                #upload(imagepath, b64img)
                up_thread = threading.Thread(target=upload, args=(imagepath, b64img,)) #上传图片
                up_thread.start()
                self.cartb.insert_one({'time':curt, 'dt':curt-self.lastt,'content':'upload car', 'imagepath':imagepath})  #日志更新
                self.carcount += 1
                if self.carcount%60 == 0:
                    print(cameraid+ '  car\t' + str(self.carcount))
                self.lastt = curt
