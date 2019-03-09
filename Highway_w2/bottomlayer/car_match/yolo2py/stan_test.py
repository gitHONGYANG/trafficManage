import sys
sys.path.append('../')
sys.path.append('../../')
import os
import cv2

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
import cfgs.config as cfg

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import visdom
from torch.autograd import Variable
from ipdb import set_trace
import torch.nn.functional as F
import datetime
import time
import uuid
import pymongo
from pymongo import MongoClient
import pprint

class similarity():
    def __init__(self, modelpath):
        self.model = torch.load(modelpath)
        self.tranform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.ColorJitter(), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def getEmbeddingformPILlist(self, pillist):
        featurelist = []

        for pilimage in pillist:
            tensorimage =  Variable(self.tranform(pilimage)).view(1, 3, 224, 224).cuda()
            feature = self.model(tensorimage)
            featurelist.append(feature)
        return featurelist

    #featurelist([id, feature, x1, y1, x2, y2, state])
    def searchmostclose(self, anchor, judgelist):
        anchorfeature = torch.div(anchor[1], torch.norm(anchor[1], 2))
        harddistance = float('Inf')
        harddistID = 0
        for i in range(len(judgelist)):
            if 'new' not in judgelist[i][6]:
                tempnegfeature = judgelist[i][1]
                # 归一化
                tempnegfeature = torch.div(tempnegfeature, torch.norm(tempnegfeature, 2))
                #tempnegfeature = torch.unsqueeze(tempnegfeature, 0)
                #set_trace()
                # print(tempnegfeature.shape)
                dis = F.pairwise_distance(anchorfeature, tempnegfeature, p=2).cpu().data.numpy()[0][0]
                if dis < harddistance:
                    harddistance = dis
                    hardnegefeature = tempnegfeature
                    harddistID = i

        return harddistID


class yoloNet():
    def __init__(self, modelpath):
        self.thresh = 0.5
        self.label = ('car', 'truck', 'trailer', 'oil')
        self.net = Darknet19()
        net_utils.load_net(modelpath, self.net)
        self.net.cuda()
        self.net.eval()
        print('Yolo init Success')

    def preprocess(self, content, method = 'nparray'):
        image = 0
        if method == 'path':
            image = cv2.imread(content)
        elif method == 'nparray':
            image = content
        else:
            print('\n\nError in preprocess\n\n')
        #yolo_utils.preprocess_test((image, none, cfg.multi_scale_inp_size), 0)将图像转换到RGB空间,resize到(320, 320),值归一化到(0,1)返回的0号元素为处理后图像,4号元素为原图拷贝
        im_data = np.expand_dims(yolo_utils.preprocess_test((image, None, cfg.multi_scale_inp_size), 0)[0], 0)
        #im_data被转换为(1,3,320,320)归一化np矩阵
        return image, im_data

    #输入path
    def getCarinfofromPic(self, content, method = 'nparray'):
        image, im_data= self.preprocess(content, method = 'nparray')
        im_data = net_utils.np_to_variable(im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
        bbox_pred, iou_pred, prob_pred = self.net(im_data)

        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, image.shape, cfg, self.thresh, size_index=0)

        roi = []
        for i in range(len(bboxes)):
            roiimage = image[bboxes[i][1]:bboxes[i][3], bboxes[i][0]:bboxes[i][2]]
            roi.append(roiimage)

        return bboxes, scores, cls_inds, image, roi

    '''
    原始程序,任然返回car
    def getCarROIpil(self, content, method = 'nparray'):
        bboxes, scores, cls_inds, ori_image, roi = self.getCarinfofromPic(content , method)
        carImage = []
        for i in range(len(bboxes)):
            pilroi = Image.fromarray(cv2.cvtColor(roi[i],cv2.COLOR_BGR2RGB))
            carImage.append(pilroi)
        return carImage, bboxes, scores, cls_inds
    '''
    #不返回car
    def getCarROIpil(self, content, method = 'nparray'):
        bboxes, scores, cls_inds, ori_image, roi = self.getCarinfofromPic(content , method)
        newbboxes , newscores, newcls_inds, newori_image, newroi = [], [], [], [], []
        carImage = []
        for i in range(len(cls_inds)):
            if cls_inds[i]!=0:
                pilroi = Image.fromarray(cv2.cvtColor(roi[i],cv2.COLOR_BGR2RGB))
                carImage.append(pilroi)
                newbboxes.append(bboxes[i])
                newscores.append(scores[i])
                newcls_inds.append(cls_inds[i])
        return carImage, newbboxes, newscores, newcls_inds


class videoreader():
    def __init__(self, videopath):
        self.cap = cv2.VideoCapture(videopath)

    def getnewframe(self, frametime, fps= 24):
        skipframe = frametime*fps
        framcount = 1
        ret, frame = self.cap.read()
        flag = True
        while framcount!=int(skipframe):
            if ret:
                flag = True
            else:
                flag = False
                break
            framcount += 1
            ret, frame = self.cap.read()
        return flag, frame

def showinVisdom(vis, frame):
    frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    frame = transforms.Resize((300, 400))(frame)
    frame = transforms.ToTensor()(frame)
    vis.image(frame)

#pool中每一元素为([(id, feature, x1, y1, x2, y2, tvstate)]),该函数根据Y轴位置将pool_last中在下一时刻离开摄像头的车辆标记为"leave_tvid", 将在下一时刻新加入的车辆标记为'enter_tvid'
def markCarleavedandnew(pool_last, pool_curent, tvid, alpha = 0, belta = 0):
    curmaxCenterY = 0
    lastminCenterY = float('Inf')
    for car in pool_curent:
        centerY = int((car[3] + car[5])/2)
        if centerY > curmaxCenterY:
            curmaxCenterY = centerY

    for car in pool_last:
        centerY = int((car[3] + car[5])/2)
        if centerY < lastminCenterY:
            lastminCenterY = centerY

    for i, car in enumerate(pool_last):
        centerY = int((car[3] + car[5])/2)
        if centerY > curmaxCenterY + alpha:
            pool_last[i][6] = 'leave_' + tvid
            print('leave a car ')

    for i, car in enumerate(pool_curent):
        centerY = int((car[3] + car[5])/2)
        if centerY < lastminCenterY - belta:
            pool_curent[i][6] = 'enter_' + tvid
            print('enter new car ')

    return pool_last, pool_curent

def test():
    y = yoloNet('../ten_thous_models/72_epoch.h5')
    s = similarity('../../checkpoints/Vgg16_CCL_NORM/epoch1_24000.pt')
    video = videoreader('../../FunctionTest/testvideo.mp4')
    flag = True
    flag, frame = video.getnewframe(0.5)
    print(flag)
    count = 1
    vis = visdom.Visdom()
    vis.close()

    curID = 1
    #[(id, feature, x1, y1, x2, y2, tvstate)]
    pool_last = []
    pool_current = []
    pool_id = []

    while flag:

        flag, frame = video.getnewframe(0.5)
        carImagelist, bboxes, scores, cls_inds = y.getCarROIpil(frame, 'nparray')
        featurelist = s.getEmbeddingformPILlist(carImagelist)
        print('time : ', str(count*0.5))

        tv = 'TV35'
        pool_current = []
        for i in range(len(carImagelist)):
            if cls_inds[i] != 0:
                carinfo = [0, featurelist[i], bboxes[i][1], bboxes[i][0], bboxes[i][3], bboxes[i][2], tv]
                pool_current.append(carinfo)

        #如果检测到车辆
        if len(pool_current)>0:
            #将上一张有车图片按Y坐标关系，标记已经离开的车，标记当前图片中新进入的车
            pool_last, pool_current = markCarleavedandnew(pool_last, pool_current, tv, 20, 20)

            #将上一张图片中未标记离开的车辆进行匹配，寻找最相似的车，并将其ID赋给curPIC中的车辆
            for i in range(len(pool_last)):
                if 'leave' not in pool_last[i][6]:
                    closeid =  s.searchmostclose(pool_last[i], pool_current)
                    pool_current[closeid][0] = pool_last[i][0]

            #对于curPIC中仍未赋ID号的车辆，赋予ID
            for i in range(len(pool_current)):
                if pool_current[i][0] == 0 and ('enter' not in pool_current[i][6]):
                    pool_current[i][0] = curID
                    curID += 1

        for i in range(len(pool_current)):
            if 'enter' in pool_current[i][6]:
                pool_current[i][0] = curID
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'ID:%d'%pool_current[i][0], (pool_current[i][3], pool_current[i][2]), font, 2, (0,255,0), 3)
                cv2.rectangle(frame,(pool_current[i][3], pool_current[i][2]), (pool_current[i][5], pool_current[i][4]), (255,0,0),3)
                curID += 1
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'ID:%d'%(pool_current[i][0]), (pool_current[i][3], pool_current[i][2]), font, 2, (0,255,0), 3)
                cv2.rectangle(frame,(pool_current[i][3], pool_current[i][2]), (pool_current[i][5], pool_current[i][4]), (255,0,0),3)

        if len(pool_current)>0:
            showinVisdom(vis, frame)

        #只在接受到新带车图像才更新
        if len(pool_current)>0:
            pool_last = pool_current

        count += 1
        if count == 1000:
            break

def buildMongo():

    client = MongoClient()
    db = client.carmatch
    carcollection = db.carinfo
    imagecollection = db.image
    #carcollection.remove({})
    #imagecollection.remove({})

    print(datetime.datetime.utcnow())

    y = yoloNet('../ten_thous_models/72_epoch.h5')
    s = similarity('../../checkpoints/Vgg16_CCL_NORM/epoch10_70000.pt')
    video = videoreader('../../FunctionTest/TV37.mp4')
    flag, frame = video.getnewframe(0.5)
    count = 1
    inittime = datetime.datetime(2018, 7, 14, 9, 0, 0, 0)
    inittimestamp = time.mktime(inittime.timetuple())
    while flag:

        t = inittimestamp + count*0.5
        t = datetime.datetime.fromtimestamp(t)
        carID = uuid.uuid1()

        print(t)
        flag, frame = video.getnewframe(0.5)
        if not flag:
            break

        carImagelist, bboxes, scores, cls_inds = y.getCarROIpil(frame, 'nparray')

        featurelist = s.getEmbeddingformPILlist(carImagelist)
        pid = -1
        cid = -1
        cameraID = 'tv37'
        picpath = '../../assests/tv37/' + str(carID) + '.jpg'
        imageid = uuid.uuid1()
        if len(cls_inds)>0:
            cv2.imwrite(picpath, frame,[int(cv2.IMWRITE_JPEG_QUALITY), 50])
            line = {"imageid":imageid, "cameraID":cameraID, "picpath":picpath, "time":t, "info":""}
            imagecollection.insert_one(line)

        for i, feature in enumerate(featurelist):
            # feature 转化为一个128维[float, float],供数据库存储
            feature = list(feature.cpu().data.numpy().flatten())
            feature = [float(f) for f in feature]
            location = bboxes[i]
            location = [int(l) for l in location]

            line = {"carid": carID, "time": t, "feature": feature, "camereid":cameraID, "location": location, "pid": pid, "cid": cid, "imgid":imageid, 'info':''}
            carcollection.insert_one(line)
            #for info in carcollection.find():
            #    pprint.pprint(info)
            #set_trace()
        count +=1
        #if count == 1000:
        #    break

    print(datetime.datetime.utcnow())


if __name__ == '__main__':
    buildMongo()
    #s = similarity('../../checkpoints/Vgg16_CCL_NORM/epoch1_24000.pt')
    #y.getCarinfofromVideo('../../FunctionTest/testvideo.mp4')
