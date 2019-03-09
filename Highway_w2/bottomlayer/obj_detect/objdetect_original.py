#本脚本涉及的外部内容很多
import numpy as np
import copy   #copy模块包括创建复合对象(包括列表、元组、字典和用户定义对象的实例)的深浅复制的函数。
from core.config import _objimgsavepath, _objrawimgsavepath, obj_tb, accident_tb, geo_loc, objmodelpath
import uuid   #生成唯一标识符
import base64  #转码
from urllib import request, parse
from utils.obj_varify import ObjValidation   #物体识别
from utils.obj_utils_original import calc_variance, contour_density_filter, morph, extra_mask_roi
from utils.region_judge import region_judge  #区域识别
import time
import os
from collections import deque   #deque 是双边队列（double-ended queue），具有队列和栈的性质，
                                # 在 list 的基础上增加了移动、旋转和增删等。
import cv2
from utils.upload_img import upload
from utils.upload_video import upload_video
import datetime
from core.config import obj_tb, car_tb
import threading

fourcc = cv2.VideoWriter_fourcc(*'XVID') #fourCC全称Four-Character Codes，代表四字符代码 (four character code),
# 它是一个32位的标示符，其实就是typedef unsigned int FOURCC;是一种独立标示视频数据流格式的四字符代码。
#因此cv2.VideoWriter_fourcc()函数的作用是输入四个字符代码即可得到对应的视频编码器。

class HighWayObjDetect:
    def __init__(self,camera_id, modelpath):
        if not hasattr(HighWayObjDetect, 'objval'):
            #HighWayObjDetect.objval = ObjValidation(modelpath)
            HighWayObjDetect.objval = ObjValidation(objmodelpath)  #objmodelpath：源：objmodelpath = '/media/assests/checkpoints/obj_resnet_29.pt'
            print('model init')

        self.camera_id = camera_id#得到摄像机号
        self.full_mask = cv2.imread('/home/highway/Highway/bottomlayer/road_mask/'+self.camera_id.lower()+'mask.png')
        self.mask, self.mask_w, self.mask_h, self.roi_w, self.roi_h = extra_mask_roi(self.full_mask)  #得到掩码区域

        self.init_frames = 200
        self.build_bg_count = 0
        self.build_bg_done = False

        self.bg_queue = deque(maxlen=15)

        self.fg_queue = deque(maxlen=20)  #初始化队列

        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=self.init_frames, varThreshold=25, detectShadows=True)
        #背景分割
        self.objoccurtimes = 0    #存取物体探测次数
        self.objoccurtimes_maxnum = 15
        self.objoccurtimes_judgenum = 10
        self.bgfg_outputboxes = []
        self.fod_box = []         #存储，self.fod_box存储的是哪些内容

        self.failcount = 0
        self.rightcount = 0
        
        #for judge weather insert an obj_info
        #fortest
        #self.db_search_startt = time.mktime(datetime.date(2018, 10, 14).timetuple())
        self.db_search_startt = time.mktime(datetime.date.today().timetuple())#起始时间
        #self.car_save_startt = time.mktime(datetime.date(2018, 7, 14).timetuple())

        self.ifsave_video = False
        self.save_video_initt = 0
        self.is_stop_car = False

        self.car_accident_time = 0   #存取car的异常时间发生的时间
        self.stop_cars = []          #存取违停的车辆

        self.save_video_type = 'obj'
        self.lastt = 0

    def build_bg(self, image):
        '''

        :param image:   摄像头原图
        :return:
        得到图像的背景
        '''
        self.image = image#原图

        self.image_road = image[self.roi_h:self.mask_h, self.roi_w:self.mask_w, :]  #得到路的区域

        if not self.build_bg_done: #背景初始化
            self.fgbg.apply(self.image_road)#self.fgbg = cv2.createBackgroundSubtractorMOG2(history=self.init_frames, varThreshold=25, detectShadows=True)

        else:#背景更新
            roiimage = self.image_road.copy()
            if len(self.bgfg_outputboxes) > 0:
                for item in self.fod_box:
                    x0 = int(item[0] - int(item[2] / 2))
                    y0 = int(item[1] - int(item[3] / 2))
                    x1 = x0 + item[2]
                    y1 = y0 + item[3]    #将图像位置的（中心点+长宽表示）转换成（对角点表示）
                    roiimage[y0:y1, x0:x1] = self.bg[y0:y1, x0:x1]
            self.fgbg.apply(roiimage)

        if self.build_bg_count == self.init_frames:
            self.build_bg_done = True
            self.bg = self.fgbg.getBackgroundImage()
            self.bg_pre = self.bg
            self.bg_queue.append(self.bg)
        else:
            #这里self.build_bg_done好像没有还原为false？？？
            self.build_bg_count += 1
            print('build bg {}/{}'.format(self.build_bg_count, self.init_frames))
            self.bg = self.fgbg.getBackgroundImage()
            self.bg_queue.append(self.bg)
    def post_fix_bg(self):
        '''
        :return:
        对背景做一定的修正
        '''
        if len(self.bgfg_outputboxes):
            fm = copy.deepcopy(self.image_road)
            for item in self.fod_box:
                x0 = int(item[0] - int(item[2]/2))
                y0 = int(item[1] - int(item[3]/2))
                x1 = x0 + item[2]
                y1 = y0 + item[3]
                self.bg_queue[-15][y0:y1, x0:x1] = self.bg_pre[y0:y1, x0:x1]
        self.bg_pre = self.bg_queue[-15]

    def extract_fg(self):#提取前景
        if self.build_bg_done:
            self.fg = cv2.absdiff(self.image_road, self.bg)   #cv2.absdiff将两幅图的差值提到另一幅图上
            self.fg = self.fg *(self.mask>0)
            gray = cv2.cvtColor(self.fg, cv2.COLOR_BGR2GRAY)
            ret, fg_raw = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)   #数为cv2.threshold()
                                                                            #这个函数有四个参数，第一个原图像，第二个进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数，常用的有：
                                                                            #• cv2.THRESH_BINARY（黑白二值）
                                                                            #• cv2.THRESH_BINARY_INV（黑白二值反转）
                                                                            #• cv2.THRESH_TRUNC （得到的图像为多像素值）
                                                                            #• cv2.THRESH_TOZERO
                                                                            #• cv2.THRESH_TOZERO_INV

            self.fg_bin = morph(fg_raw,3,8)            #形态学处理
            self.post_fix_bg()                         #背景修正

    def find_coutours_update_boxes(self, fgimage):
        '''
        :param fgimage:前景图
        :return:
        功能：找出轮廓，记录位置
        '''
        (_, cnts, _) = cv2.findContours(fgimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS) #轮廓

        if self.objoccurtimes == 0:     #self.objoccurtimes表示box更新的次数，即物体探测的次数
            self.boxes_pre = []
        org_h = self.mask_h - self.roi_h
        boxes, cnt_poten = contour_density_filter(cnts, 150, 160000, org_h)   #密度筛选，面积筛选 box记录
        # 了轮廓的位置（中心点+长宽） cnt_poten记录了筛选后的轮廓

        '''
        for box in boxes:
            x0 = int(box[0]-box[2]/2)
            y0 = int(box[1]-box[3]/2)
            x1 = int(box[0]+box[2]/2)
            y1 = int(box[1]+box[3]/2)
            cv2.rectangle(self.vis, (x0,y0), (x1,y1), (255, 0, 0), 1)
        '''
        #比较跟新后的box与以前的box
        for box in boxes:
            for bp in self.boxes_pre:
                # adaptive threshold by the y coord
                pos_range = int((box[1] / org_h) * 8) + 1
                size_range = int((box[1] / org_h) * 6) + 1
                if (abs(box[0] - bp[0]) < pos_range) & (abs(box[1] - bp[1]) < pos_range) and ((abs(box[2] - bp[2]) < size_range) & (abs(box[3] - bp[3]) < size_range)):#如果位置变动不大
                    bp[4] = bp[4] + 1   #高度加一
                    box[4] = bp[4]

        self.boxes_pre = boxes
        if self.objoccurtimes == self.objoccurtimes_maxnum:
            self.bgfg_outputboxes = [box for box in self.boxes_pre if box[4]>self.objoccurtimes_judgenum]
            self.objoccurtimes = 0
        else:
            self.objoccurtimes += 1

        return self.bgfg_outputboxes

    def if_insert(self, t, location,cam_id):
        '''
        :param t: 时间
        :param location:  位置
        :param cam_id:   camera_id
        :return:
        功能：是否插入数据库
        感觉有些操作没用
        '''
        #print('if insert')
        #print('db_search_start\t', str(self.db_search_startt))
        #print('judget\t', str(t-3*60))
        if self.db_search_startt > t-3*60:   #时间间隔未到3分钟
            return False
        
        centerx = (location[0] + location[2])/2
        centery = (location[1] + location[3])/2     #得到中心点位置
        startt = self.db_search_startt
        endt = t
        where = {'time':{'$gt':startt, '$lt':endt}, 'cameraid': cam_id}
        obj_infos = list(obj_tb.find(where))   #找到对应obj的信息
        
        ifinsert = True
        for obj_info in obj_infos:
            org_h = 600
            pos_range = int((location[3] + location[1])/2/org_h * 8) + 1

            loca_his = obj_info['location']
            centerx_his = (loca_his[0] + loca_his[2])/2
            centery_his = (loca_his[1] + loca_his[3])/2
                
            if abs(centerx_his - centerx) < pos_range or abs(centerx_his - centerx) < pos_range:    #当位置变化很小
                ifinsert = False
                break
        
        if ifinsert:
            self.db_search_startt = t
        
        #print('if insert\t', str(ifinsert), '\tself.db_search_startt\t', str(self.db_search_startt))
        return ifinsert

    def judge_car_accident(self, cameraid, nowtime):
        '''
        :param cameraid:
        :param nowtime:  目前的时间
        :return:
        功能：判断是否发生了事故
        '''
        initt = time.time()

        t = nowtime
        where = {'time':{'$gt':t-2, '$lt':t}, 'cameraid':cameraid}  #把时间提前两秒
        self.rec_cars = list(car_tb.find(where))  #找到车的信息

        if not self.rec_cars:     #信息为空，没有停车
            print(self.camera_id, 'stop cars\n[]')
            return []

        ts = []
        for car in self.rec_cars:
            ts.append(car['time'])
        ts = sorted(ts)  #对找到的信息按时间排序
        recentt = ts[-1] #最近的时间

        temp = []
        for car in self.rec_cars:
            if car['time'] == recentt:
                temp.append(car)   #将最近时间的车的信息读入
        self.rec_cars = temp

        where = {'time':{'$gt':t-22, '$lt':t-2}, 'cameraid':cameraid}  #将时间提前22秒
        search_cars = list(car_tb.find(where))
        
        stop_cars = []
        for car in self.rec_cars:
            anc_location = car['location']
            anc_centery = int((anc_location[3] - anc_location[1])/2 + anc_location[1])
            anc_centerx = int((anc_location[2] - anc_location[0])/2 + anc_location[0])
            anchor_w = int(anc_location[2] - anc_location[0])
            anchor_h = int(anc_location[3] - anc_location[1])
            shift_w = int(anchor_w/8)
            shift_h = int(anchor_h/8)    #得到各种位置形状信息已即偏移量的阈值
            
            count = 0
            for sear_car in search_cars:
                location = sear_car['location']
                centery = int((location[3] - location[1])/2 + location[1])
                centerx = int((location[2] - location[0])/2 + location[0])
                if abs(centerx-anc_centerx)<shift_w and abs(centery-anc_centery)<shift_h:  #判断偏移量的大小
                    count += 1
            if count>10:    #次数超过十次，则判断车停在路上
                stop_cars.append(anc_location)
        print(self.camera_id,'stop cars\n', str(stop_cars), '\t', 'judge cost time\t', str(time.time() - initt))
        
        return stop_cars

    def accident_ifinsert(self, t, location,cam_id):
        '''

        :param t: 时间
        :param location: 位置
        :param cam_id:  camera_id
        :return:
        功能：判断是否将事故信息写入数据库
        '''
        centerx = (location[0] + location[2])/2
        centery = (location[1] + location[3])/2
        startt = self.db_search_startt
        endt = t
        where = {'time':{'$gt':startt, '$lt':endt}, 'cameraid': cam_id}  #找到附近时间存储的车的信息
        obj_infos = list(accident_tb.find(where))
        
        ifinsert = True
        #这里没有对obj_infos做空集处理
        for obj_info in obj_infos:
            org_h = 600
            pos_range = int((location[3] - location[1])/10)

            loca_his = obj_info['location']
            centerx_his = (loca_his[0] + loca_his[2])/2
            centery_his = (loca_his[1] + loca_his[3])/2
                
            if abs(centerx_his - centerx) < pos_range and abs(centerx_his - centerx) < pos_range:  #表明数据已经写入
                ifinsert = False
                break
        
        return ifinsert
    
    def draw_rect(self, nowtime):
        '''
        :param nowtime:
        :return:
        将违规车存入数据库，并将区域框出来
        '''
        self.save_image = self.image.copy()
        self.bgfg_outputboxes = self.find_coutours_update_boxes(self.fg_bin)   #fig_bin存储前景图

        org_var = copy.deepcopy(self.image_road)   #复制路图
        org_var_roi = org_var * self.mask
        var_gray = cv2.cvtColor(org_var_roi, cv2.COLOR_BGR2GRAY)

        mask_var = calc_variance(var_gray,size=5, stride=2)  #计算方差
        mask_var = cv2.resize(mask_var,(self.image_road.shape[1],self.image_road.shape[0]))

        rawimagepath = _objrawimgsavepath + self.camera_id + '_' + str(nowtime) + '.jpg'  #确定存储路径

        #get stop cars in this pic
        img_upload = False
        if nowtime > self.car_accident_time + 20 and self.ifsave_video == False:
            self.car_accident_time = nowtime
            self.stop_cars = self.judge_car_accident(self.camera_id, nowtime)
            for car in self.stop_cars:
                y = int((car[1] + car[3]) /2)
                y = int((1080-y)/1080 * 250 + 20)
                cameraid = self.camera_id    #得到基础信息
                pos_1, pos_2 = geo_loc[cameraid]  #camera的位置
                pos_2 += y
                if pos_2>1000:
                    pos_2 -= 1000
                    pos_1 += 1       #舍弃了x的信息
                position = 'K' + str(pos_1) + '+' + str(pos_2)
                objid = str(uuid.uuid1())  #获得唯一id
                info = ''
                point = (int((car[0] + car[2])/2), car[3])
                info = region_judge(point, cameraid.lower())
                saferank = 2
                if info == 'edge':
                    saferank = 2
                elif info == 'road':
                    saferank = 0               

                acciinfo = {'imagepath':rawimagepath, 'cameraid':cameraid, 'time':nowtime, 'location':car, 'position':position, 'saferank':saferank,'objid':objid, 'state':0, 'info':info}
                
                ifinsert = self.accident_ifinsert(nowtime, car, cameraid)   #判断是否加入数据库
                if ifinsert:
                    img_upload = True
                    self.ifsave_video = True
                    self.save_video_type = 'car'

                    self.videoname = '/media/assests/ObjVideos/' + cameraid + '_' + str(nowtime) + '.avi'
                    self.savevideo = cv2.VideoWriter(self.videoname, fourcc, 20.0, (1366, 768))     #视频存储
                    acciinfo['videopath'] = '/media/assests/Objvideos/' + cameraid + '_' + str(nowtime) + '.avi'
                    accident_tb.insert_one(acciinfo)
                    obj_tb.insert_one(acciinfo)
        
        if self.ifsave_video == True and nowtime < self.car_accident_time + 60:
            for car in self.stop_cars:
                cv2.rectangle(self.save_image, (car[0], car[1]), (car[2], car[3]), (0, 0, 255), 2)
            
        for box in self.bgfg_outputboxes:
            if self.ifsave_video == True:
                break
            #break
            x0 = max(int(box[0] - box[2] / 2)-10,0)
            y0 = max(int(box[1] - box[3] / 2)-10,0)
            x1 = min(int(box[0] + box[2] / 2)+10,self.fg_bin.shape[1])
            y1 = min(int(box[1] + box[3] / 2)+10,self.fg_bin.shape[0])   #得到对角点的坐标
            objimage = self.image_road[y0:y1, x0:x1]
            if ((mask_var[y0:y1, x0:x1]>0.05).sum() / (box[2]*box[3])>0.75) and (HighWayObjDetect.objval.varify_cv(objimage)):   #物体探测
                #print('mask var:', (mask_var[y0:y1, x0:x1]>0.05).sum() / (box[2]*box[3]))
                print(self.camera_id, ' find some thing\t', str(datetime.datetime.now()))
                self.rightcount += 1
                #print('true', str(self.rightcount))
                x0_org = x0 + self.roi_w
                y0_org = y0 + self.roi_h
                x1_org = x1 + self.roi_w
                y1_org = y1 + self.roi_h
                cv2.rectangle(self.save_image, (x0_org, y0_org), (x1_org, y1_org), (0, 0, 255), 2)
                
                location = [x0_org, y0_org, x1_org, y1_org]
                cameraid = self.camera_id
                y = (location[1] + location[3])/2
                y = int((1080-y)/1080 * 250 + 20)
                pos_1, pos_2 = geo_loc[cameraid]
                pos_2 += y
                if pos_2>1000:
                    pos_2 -= 1000
                    pos_1 += 1
                position = 'K' + str(pos_1) + '+' + str(pos_2)

                area = (x1_org-x0_org)*(y1_org-y0_org)  #确定问题区域
                saferank = 2      #确定安全等级
                if area > 10000:
                    saferank = 0
                elif area > 2500 and saferank>1:
                    saferank = 1

                objid = str(uuid.uuid1())
                info = ''
                #state: 0-wait for handling, 1-had handled, 2-ignore
                objinfo = {'imagepath':rawimagepath, 'cameraid':cameraid, 'time':nowtime, 'location':location, 'position':position, 'saferank':saferank,'objid':objid, 'state':0, 'info':info}
                print(objinfo) 
                ifinsert = self.if_insert(nowtime, location, cameraid)
                if ifinsert:
                    img_upload = True
                    self.ifsave_video = True
                    self.save_video_type = 'obj'

                    self.videoname = '/media/assests/ObjVideos/' + cameraid + '_' + str(nowtime) + '.avi'
                    self.savevideo = cv2.VideoWriter(self.videoname, fourcc, 20.0, (1366, 768))
                    objinfo['videopath'] = '/media/assests/Objvideos/' + cameraid + '_' + str(nowtime) + '.avi'
                    obj_tb.insert_one(objinfo)
                    print('-------start video--------\n')
            else:      #没探测到物体
                self.failcount += 1
                self.bgfg_outputboxes.remove(box)
                #cv2.rectangle(self.vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
                #print('false ', str(self.failcount))

        if img_upload:
            lowppi_frame = cv2.resize(self.save_image, (800, 450))
            ret, buffer = cv2.imencode('.jpg', lowppi_frame)   #上传图片
            if ret:
                b64img = base64.b64encode(buffer)
                up_thread = threading.Thread(target=upload, args=(rawimagepath, b64img,))
                #upload(rawimagepath, b64img)
                up_thread.start()
                print('upload image\t', rawimagepath)

        if self.ifsave_video:
            lowppi_frame = cv2.resize(self.save_image, (1366, 768))
            self.savevideo.write(lowppi_frame)   #将图片一帧帧的传入
            #print('write frame')
            if nowtime - self.db_search_startt > 3*60-30 and self.save_video_type == 'obj':   #探测的异物
                self.ifsave_video = False
                self.savevideo.release()
                videoname = self.videoname.split('/')[-1]
                initt = time.time()
                

                up_thread = threading.Thread(target=upload_video, args=(self.videoname, videoname,))
                up_thread.start()
                #upload_video(self.videoname, videoname)
                print('upload \t', videoname, '\t', str(time.time() - initt))
                print('--------end video---------\n')

            if nowtime - self.car_accident_time > 3*60-30 and self.save_video_type == 'car':  #探测到的是车
                self.ifsave_video = False
                self.savevideo.release()
                
                videoname = self.videoname.split('/')[-1]
                initt = time.time()
                up_thread = threading.Thread(target=upload_video, args=(self.videoname, videoname,))
                #upload_video(self.videoname, videoname)
                up_thread.start()
                print('upload \t', videoname, '\t', str(time.time() - initt))
                print('--------end video---------\n')
            
        
        self.lastt = nowtime
        
        #cv2.imshow('save_image', self.save_image)

    def step(self, frame, nowtime):
        #print(nowtime)
        self.build_bg(frame)
        self.extract_fg()
        #self.visualization()
        if self.build_bg_done:
            self.draw_rect(nowtime)

    def visualization(self):  #图片可视化
        if self.build_bg_done:
            cv2.imshow('fg', self.fg)
            cv2.imshow('bg_pre', self.bg_pre)
            cv2.imshow('fg_bin', self.fg_bin)
            cv2.imshow('road', self.image_road)

            cv2.waitKey(1)


if __name__ == '__main__' :

    test_video = './videofortest/test_TV52_01.mp4'
    cap = cv2.VideoCapture(test_video)
    # process per process_count
    process_count = 20
    count = 0
    detect_obj = HighWayObjDetect('tv52')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % process_count == 0:
            detect_obj.step(frame)

        count += 1
