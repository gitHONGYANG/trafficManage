import os
from pymongo import MongoClient
import numpy as np
import pandas as pd
import shutil
import time
from carmatch.geo_config import geo_dis, segments
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable
from config import tvconfig_history_tb

#TV15没进服务区，TV17已出服务区，需检查TV16;TV35没进停车区，TV36已出停车区;TV50没进服务区, TV51已出服务区;TV77/TV78 - 灵山服务区
parks = [16, 35, 50, 78]

#tv10可见,tv11不可见;TV18缺失,tv19不可见;tv27在, tv28不确定, tv29不可见;tv31可见, tv32向左驶出(可见), tv33不可见;tv43可见, tv44 不可见;tv55可见, tv56不可见;tv59可见, tv60向左驶出(可见), tv61不可见;tv74可见, tv76不可见
leaves = [11, 19, 29, 33, 44, 56, 61, 76]
modelpath = '/home/highway/model/CCL_epoch10_305000.pt'

class Tv():
    def setvalue(self, isforwardpark, isforwardleave, isbackwardpark, isbackwardleave, dis_forward, dis_backward):
        self.isforwardpark = isforwardpark
        self.isforwardleave = isforwardleave
        self.isbackwardpark = isbackwardpark
        self.isbackwardleave = isbackwardleave
        self.dis_forward = dis_forward
        self.dis_backward = dis_backward
        self.dis = float('Inf')
        self.t = float('Inf')
    def setdis(self, dis):
        self.dis = dis
    def sett(self, t):
        self.t = t

def gettvconfig(t):
    '''
    tvconfig_tv = MongoClient('mongodb://192.168.6.188:27017')['highway']['tvconfig']
    tvs = list(tvconfig_tv.find({'name':'w2'})) + list(tvconfig_tv.find({'name':'w3'})) + list(tvconfig_tv.find({'name':'w4'})) + list(tvconfig_tv.find({'name':'w5'}))
    tvs_active = []
    for tv in tvs:
        tvs_active += tv['tvs']
    tvs_active.sort(key = lambda x: int(x.split('V')[-1]))
    '''
    curt = t
    where = {'time':{'$gt':t}}
    tvs_active = list(tvconfig_history_tb.find(where))[0]['tvconfig']

    return tvs_active

def getactive_tvinfos(tvs_active):
    active_tvinfos = {}
    print('activate tvs\t', str(tvs_active))

    for i, tv in enumerate(tvs_active):
        info = Tv()
        if i==0:
            isbackwardpark = False
            isbackwardleave = False
            dis_backward = float('Inf')
        else :
            loc_distance = geo_dis[tvs_active[i]]
            back_distance = geo_dis[tvs_active[i-1]]
            dis_backward = (back_distance[0] - loc_distance[0]) * 1000 + back_distance[1] - loc_distance[1]
            isbackwardpark = False
            isbackwardleave = False
            for park in parks:
                if int(tv.split('V')[-1])>park  and int(tvs_active[i-1].split('V')[-1])<=park:
                    isbackwardpark = True
            for leave in leaves:
                if int(tv.split('V')[-1])>=leave and int(tvs_active[i-1].split('V')[-1])<leave:
                    isbackwardleave = True

        if i == len(tvs_active) - 1:
            isforwardpark = False
            isforwardleave = False
            dis_forward = float('Inf')
        else:
            loc_distance = geo_dis[tvs_active[i]]
            forward_distance = geo_dis[tvs_active[i+1]]
            dis_forward = (forward_distance[0] - loc_distance[0]) * 1000 + forward_distance[1] - loc_distance[1]
            isforwardpark = False
            isforwardleave = False
            for park in parks:
                if int(tv.split('V')[-1])<=park and int(tvs_active[i+1].split('V')[-1])>park:
                    isforwardpark = True
            for leave in leaves:
                if int(tv.split('V')[-1])<leave and int(tvs_active[i+1].split('V')[-1])>=leave:
                    isforwardleave = True
        info.setvalue(isforwardpark, isforwardleave, isbackwardpark, isbackwardleave, dis_forward, dis_backward)
        active_tvinfos[tv] = info

    return active_tvinfos


#tvs_active = gettvconfig()
#active_tvinfos = getactive_tvinfos(tvs_active)

#for key in active_tvinfos.keys():
    #print(key, '\tbackward\t', str(active_tvinfos[key].dis_backward), 'forward\t', str(active_tvinfos[key].dis_forward))
    #print('isforward leave\t', str(active_tvinfos[key].isforwardleave), '\tifbackward leave\t', str(active_tvinfos[key].isbackwardleave))
    #print('isforward park\t', str(active_tvinfos[key].isforwardpark), '\tifbackward park\t', str(active_tvinfos[key].isbackwardpark))

class BaseNetwork(nn.Module):

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

class similarity():
    def __init__(self):
        self.model = BaseNetwork('Vgg16')
        self.model.load_state_dict(torch.load(modelpath))
        self.model.cuda()
        #self.model = torch.load(modelpath).cuda()
        self.transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #return feature by list
    def get_pil_feat(self, pilimage):
        starttime = time.time()
        tensorimage = Variable(self.transform(pilimage)).view(1, 3, 224, 224).cuda()
        feature = self.model(tensorimage)
        feature = list(feature.cpu().data.numpy().flatten()*1000)
        feature = [int(f) for f in feature]
        return feature

#for tv in active_tvinfos.keys():
#    info = active_tvinfos[tv]
    #print('------%s-----' %tv)
    #print('isforwardpark\t', str(info.isforwardpark))
    #print('isbackwardpark\t', str(info.isbackwardpark))
    #print('isforwardleave\t', str(info.isforwardleave))
    #print('isbackwardleave\t', str(info.isbackwardleave))
    #print('dis_forward\t', str(info.dis_forward))
    #print('dis_backward\t', str(info.dis_backward))    

class MatchCar():
    def __init__(self):
        self.carid_anc = ''
        self.camid_anc = ''
        self.feat_anc = ''
        self.disbase_anc = 0
        self.t_anc = 0
        self.similarity = similarity()

    def reinit(self):
        self.carid_anc = ''
        self.camid_anc = ''
        self.feat_anc = ''
        self.disbase_anc = 0
        self.t_anc = 0

    def set_rootdf(self, df):
        self.rootdf = df

    def get_ancinfo(self, pilimage, t_anc, camid_anc):

        #for i , camid in enumerate(tvs_active):
        #   if camid == self.camid_anc:
        #       self.index_camid = i
        self.t_anc = t_anc
        self.camid_anc = camid_anc
        self.feat_anc = self.similarity.get_pil_feat(pilimage)

        self.state_forwardsearch = 0  #0搜索未结束, 1搜索结束
        self.state_backwardsearch = 0 #0搜索未结束, 1搜索结束
        self.forward_breakpoint = self.camid_anc
        self.backward_breakpoint = self.camid_anc
        self.forward_breakpoint_t = self.t_anc
        self.backward_breakpoint_t = self.t_anc

    def search_base(self, camid_tar, t_start, t_end, num):
        df_tar = self.rootdf[(self.rootdf['cameraid'] == camid_tar) & (self.rootdf['time']>=t_start) & (self.rootdf['time']<t_end)].copy()
        #closest_car =

        df_tar['distance'] = df_tar['feature'].apply(lambda  x:np.linalg.norm(np.array(x) - self.feat_anc))
        df_tar = df_tar.sort_values(by = 'distance')
        closestcars = df_tar[['distance', 'carid', 'imagepath', 'time', 'cameraid', 'location', 'position']].values

        return closestcars[:num]

    def search_anchorseg(self):
        #获得anchor segment(连续摄像头)
        anc_geosegment = []
        for segment in segments:
            if int(self.camid_anc.split('V')[-1]) in segment:
                anc_geosegment = segment
        anc_segment = []
        for camid in anc_geosegment:
            if 'TV'+str(camid) in self.tvs_active:
                anc_segment.append(camid)

        anc_index = 0
        for i, tv in enumerate(anc_segment):
            if tv == int(self.camid_anc.split('V')[-1]):
                anc_index = i
        search_segment_forward = []

        for i in range(anc_index + 1, len(anc_segment)):
            camid = 'TV' + str(anc_segment[i])
            if self.active_tvinfos[camid].isforwardpark == True:
                search_segment_forward.append(camid)
                break
            else:
                search_segment_forward.append(camid)
        if self.active_tvinfos[self.camid_anc].isforwardpark == True:
            search_segment_forward = []

        search_segment_backward = []
        for i in range(anc_index - 1, -1, -1):
            camid = 'TV' + str(anc_segment[i])
            if self.active_tvinfos[camid].isbackwardpark == True:
                search_segment_backward.append(camid)
                break
            else:
                search_segment_backward.append(camid)
        if self.active_tvinfos[self.camid_anc].isbackwardpark == True:
            search_segment_backward = []

        release_segment_forward = []
        release_segment_backward = []

        flag = 0
        tv_judge = 'TV0'
        for tv in self.tvs_active:
            if flag == 1:
                release_segment_forward.append(tv)
            if search_segment_forward:
                tv_judge = search_segment_forward[-1]
            else:
                tv_judge = self.camid_anc
            if tv == tv_judge:
                flag = 1

        flag = 0
        tv_judge = 'TV0'
        temp = self.tvs_active.copy()[::-1]
        for tv in temp:
            if flag == 1:
                release_segment_backward.append(tv)
            if search_segment_backward:
                tv_judge = search_segment_backward[-1]
            else:
                tv_judge = self.camid_anc
            if tv == tv_judge:
                flag =1


        #print('release_segment_forward\t', str(release_segment_forward))
        #print('release_segment_backward\t', str(release_segment_backward))
        return search_segment_forward, search_segment_backward, release_segment_forward, release_segment_backward

    def get_disbase(self):
        dis = 0
        anchor_cars = self.search_base(self.camid_anc, self.t_anc-5, self.t_anc + 5, 3)
        if len(anchor_cars) == 1:
            dis = 10000
        else:
            for car in anchor_cars:
                dis += car[0]
        search_segment_forward, search_segment_backward, release_segment_forward, release_segment_backward = self.search_anchorseg()
        ancseg_forward = self.search_forward(search_segment_forward, self.camid_anc, self.t_anc)
        ancseg_backward = self.search_backward(search_segment_backward, self.camid_anc, self.t_anc)

        num_diss = 0
        for camid in search_segment_forward:
            if self.active_tvinfos[camid].dis != float('Inf'):
                dis += self.active_tvinfos[camid].dis
                num_diss += 1
        for camid in search_segment_backward:
            if self.active_tvinfos[camid].dis != float('Inf'):
                dis += self.active_tvinfos[camid].dis
                num_diss += 1

        self.disbase_anc = dis/(num_diss+ len(anchor_cars)-1)
        print('disbase_anc\t', str(self.disbase_anc))
        return ancseg_forward, ancseg_backward, release_segment_forward, release_segment_backward, anchor_cars

    def search_forward(self, segment, cam_cur, t_cur):
        index_start = 0
        index_end = 0

        print('\n\n---search forward---')
        #print('segment\t', str(segment))
        #print('cam_cur\t', str(cam_cur))

        if len(segment) == 0 or self.state_forwardsearch == 1:
            return []
        for i, camid in enumerate(self.tvs_active):
            if cam_cur == camid:
                index_start = i
            if segment[-1] == camid:
                index_end = i

        answer = []
        #print('index_start\t', str(index_start))
        #print('index_end-1\t', str(index_end-1))
        for i in range(index_start, index_end):
            camid_cur = self.tvs_active[i]
            camid_tar = self.tvs_active[i+1]

            self.forward_breakpoint = camid_cur
            self.forward_breakpoint_t = t_cur

            info_cur = self.active_tvinfos[camid_cur]
            dis_cur = 0
            if info_cur.isforwardpark == False and info_cur.isforwardleave == False:
                t_start = t_cur + info_cur.dis_forward*3.6/150
                t_end = t_cur + info_cur.dis_forward*3.6/30
                closecars = self.search_base(camid_tar, t_start, t_end, 2)

                #print('t_start\t', str(t_start))
                #print('t_end\t', str(t_end))
                #print('cam_tar\t', camid_tar)

                if len(closecars)==0:
                    self.state_forwardsearch = 1
                    print(camid_tar, '\tnot fonud any car')
                    return answer

                for car in closecars:
                    answer.append(car)

                tempt = 0
                temp_dis = 0
                for car in closecars:
                    tempt += car[3]/len(closecars)
                    temp_dis += car[0]/len(closecars)
                t_cur = tempt
                dis_cur = temp_dis
                self.active_tvinfos[camid_tar].setdis(dis_cur)
                self.active_tvinfos[camid_tar].sett(t_cur)
                self.forward_breakpoint = camid_tar
                self.forward_breakpoint_t = t_cur

            elif info_cur.isforwardpark == True:

                print('\n\nforward_park')
                #第一次搜索
                t_start = t_cur + info_cur.dis_forward*3.6/150
                t_end = t_cur + info_cur.dis_forward*3.6/30
                closecars = self.search_base(camid_tar, t_start, t_end, 2)

                temp_dis = 0
                tempt = 0
                first_search = True
                for car in closecars:
                    temp_dis += car[0]/len(closecars)
                    tempt += car[3]/len(closecars)
                dis_cur = temp_dis
                t_cur = tempt
                if dis_cur == 0 or dis_cur > self.disbase_anc*1.6:
                    first_search = False
                else:
                    for car in closecars:
                        answer.append(car)
                    self.active_tvinfos[camid_tar].setdis(dis_cur)
                    self.active_tvinfos[camid_tar].sett(t_cur)
                    self.forward_breakpoint = camid_tar
                    self.forward_breakpoint_t = t_cur

                #第二次搜索
                #若first_search == False, 则在服务站未找到
                t_park = 0
                check_tv = 'TV0'
                if first_search == False:
                    if camid_tar == 'TV16' or camid_tar == 'TV17':
                        check_tv = 'TV17'
                    elif camid_tar == 'TV35' or camid_tar == 'TV36':
                        check_tv = 'TV36'
                    elif camid_tar == 'TV51' or camid_tar == 'TV52':
                        check_tv = 'TV52'
                    elif camid_tar == 'TV78' or camid_tar == 'TV79':
                        check_tv = 'TV79'
                    print(camid_tar)
                    check_tv_geo = geo_dis[check_tv]

                    print('park search')
                    print('camid_cur\t', camid_cur)
                    print('check_tv\t', check_tv)
                    t_start = t_cur + (1000 * (geo_dis[camid_cur][0] - check_tv_geo[0]) + geo_dis[camid_cur][1] - check_tv_geo[1])/150
                    t_end = t_start + 1800
                    while True:
                        print(t_start)
                        print(t_end)
                        closecars = self.search_base(check_tv, t_start, t_end, 2)
                        if len(closecars) == 0:
                            print('park search end, not found')
                            break

                        dis_temp = 0
                        t_temp = 0
                        for car in closecars:
                            dis_temp += car[0]/len(closecars)
                            t_temp += car[3]/len(closecars)
                        if dis_temp < self.disbase_anc * 1.6:
                            t_park = t_temp
                            break
                        t_start = t_end
                        t_end = t_start + 1800

                #print('----t_park----', t_park)
                if t_park == 0 and first_search == False:
                    self.state_forwardsearch = 1
                    return answer

                if t_park != 0 and camid_tar != check_tv:
                    print('3 search')

                    t_start = t_park - 2000*3.6/30 -60
                    t_end = t_park - 1000*3.6/150
                    closecars = self.search_base(camid_tar, t_start, t_end, 2)

                    temp_dis = 0
                    tempt = 0
                    for car in closecars:
                        temp_dis += car[0]/len(closecars)
                        tempt += car[3]/len(closecars)
                    t_cur = tempt
                    dis_cur = temp_dis
                    if dis_cur < self.disbase_anc*2 and dis_cur!=0:
                        for car in closecars:
                            answer.append(car)
                        self.active_tvinfos[camid_tar].setdis(dis_cur)
                        self.active_tvinfos[camid_tar].sett(t_cur)
                        self.forward_breakpoint = camid_tar
                        self.forward_breakpoint_t = t_cur

                print('forward breakpoint\t', self.forward_breakpoint)

            elif info_cur.isforwardleave == True:
                print('foward_leave')

                t_start = t_cur + info_cur.dis_forward*3.6/150
                t_end = t_cur + info_cur.dis_forward*3.6/30
                #print('t_start\t', str(t_start))
                #print('t_end\t', str(t_end))
                #print('cam_tar\t', camid_tar)

                closecars = self.search_base(camid_tar, t_start, t_end, 2)
                if len(closecars)==0:
                    self.state_forwardsearch = 1
                    print('no car found')
                    return answer
                else:
                    tempt = 0
                    temp_dis = 0
                    for car in closecars:
                        temp_dis += car[0]/len(closecars)
                        tempt += car[3]/len(closecars)
                    t_cur = tempt
                    dis_cur = temp_dis
                if dis_cur > self.disbase_anc * 1.6:
                    self.state_forwardsearch = 1
                    return answer
                else:
                    for car in closecars:
                        answer.append(car)
                    self.active_tvinfos[camid_tar].setdis(dis_cur)
                    self.active_tvinfos[camid_tar].sett(t_cur)
                    self.forward_breakpoint = camid_tar
                    self.forward_breakpoint_t = t_cur
            #print(active_tvinfos[camid_tar].dis)
        return answer

    def search_backward(self, segment, cam_cur, t_cur):
        index_start = 0
        index_end = 0

        print('\n\n---search backward---')
        #print('segment\t', str(segment))
        #print('cam_cur\t', str(cam_cur))

        if len(segment) == 0 or self.state_backwardsearch == 1:
            return []
        for i, camid in enumerate(self.tvs_active):
            if cam_cur == camid:
                index_start = i
            if segment[-1] == camid:
                index_end = i

        answer = []
        #print('index_start\t', str(index_start))
        #print('index_end\t', str(index_end))
        for i in range(index_start, index_end, -1):
            camid_cur = self.tvs_active[i]
            camid_tar = self.tvs_active[i - 1]

            self.backward_breakpoint = camid_cur
            self.backward_breakpoint_t = t_cur

            info_cur = self.active_tvinfos[camid_cur]
            dis_cur = 0
            if info_cur.isbackwardpark == False and info_cur.isbackwardpark == False:
                t_start = t_cur + info_cur.dis_backward * 3.6 / 30
                t_end = t_cur + info_cur.dis_backward * 3.6 / 150
                closecars = self.search_base(camid_tar, t_start, t_end, 2)

                #print('t_start\t', str(t_start))
                #print('t_end\t', str(t_end))
                #print('cam_tar\t', camid_tar)

                if len(closecars) == 0:
                    self.state_backwardsearch = 1
                    print(camid_tar, '\tnot fonud any car')
                    return answer

                for car in closecars:
                    answer.append(car)

                tempt = 0
                temp_dis = 0
                for car in closecars:
                    tempt += car[3] / len(closecars)
                    temp_dis += car[0] / len(closecars)
                t_cur = tempt
                dis_cur = temp_dis
                self.active_tvinfos[camid_tar].setdis(dis_cur)
                self.active_tvinfos[camid_tar].sett(t_cur)
                self.backward_breakpoint = camid_tar
                self.backward_breakpoint_t = t_cur

            elif info_cur.isbackwardpark == True:
                print('\nbackward_park')
                # 第一次搜索

                t_start = t_cur + info_cur.dis_backward * 3.6 / 30
                t_end = t_cur + info_cur.dis_backward * 3.6 / 150
                closecars = self.search_base(camid_tar, t_start, t_end, 2)
                #print('t_star\t', str(t_start))
                #print('t_end\t', str(t_end))

                dis_temp = 0
                tempt = 0
                first_search = True
                for car in closecars:
                    dis_temp += car[0] / len(closecars)
                    tempt += car[3] / len(closecars)
                if dis_temp == 0 or dis_temp > self.disbase_anc * 1.6:
                    first_search = False
                else:
                    for car in closecars:
                        answer.append(car)
                    self.active_tvinfos[camid_tar].setdis(dis_cur)
                    self.active_tvinfos[camid_tar].sett(t_cur)
                    self.backward_breakpoint = camid_tar
                    self.backward_breakpoint_t = t_cur

                # 第二次搜索
                # 若first_search == False, 则在服务站未找到
                t_park = 0
                if first_search == False:
                    t_end = t_cur + info_cur.dis_backward * 3.6 / 30
                    t_start = t_start - 1800
                    while True:
                        closecars = self.search_base(camid_tar, t_start, t_end, 2)
                        if len(closecars) == 0:
                            print('park search end, not found')
                            self.state_backwardsearch = 1
                            return answer

                        dis_temp = 0
                        t_temp = 0
                        for car in closecars:
                            dis_temp += car[0] / len(closecars)
                            t_temp += car[3] / len(closecars)
                        if dis_cur < self.disbase_anc * 1.6:
                            t_cur = t_temp
                            dis_cur = dis_temp

                            self.active_tvinfos[camid_tar].setdis(dis_cur)
                            self.active_tvinfos[camid_tar].sett(t_cur)
                            self.backward_breakpoint = camid_tar
                            self.backward_breakpoint_t = t_cur
                            break

                        t_end = t_start
                        t_start = t_end - 1800

            elif info_cur.isbackwardleave == True:
                print('backward_leave')

                t_start = t_cur + info_cur.dis_forward * 3.6 / 30
                t_end = t_cur + info_cur.dis_forward * 3.6 / 150
                #print('t_start\t', str(t_start))
                print('t_end\t', str(t_end))
                #print('cam_tar\t', camid_tar)

                closecars = self.search_base(camid_tar, t_start, t_end, 2)
                if len(closecars) == 0:
                    self.state_backwardsearch = 1
                    print('no car found')
                    return answer
                else:
                    tempt = 0
                    temp_dis = 0
                    for car in closecars:
                        temp_dis += car[0] / len(closecars)
                        tempt += car[3] / len(closecars)
                    t_cur = tempt
                    dis_cur = temp_dis
                if dis_cur > self.disbase_anc * 1.6:
                    self.state_backwardsearch = 1
                    return answer
                else:
                    for car in closecars:
                        answer.append(car)
                    self.active_tvinfos[camid_tar].setdis(dis_cur)
                    self.active_tvinfos[camid_tar].sett(t_cur)
                    self.forward_breakpoint = camid_tar
                    self.forward_breakpoint_t = t_cur
        return answer

    def searchcar(self, pilimage, t_anc, camid_anc, df):
        #self.tvs_active = gettvconfig(t_anc)
        self.tvs_active = ['TV%d' %i for i in range(31, 64)]
        self.tvs_active.remove('TV55')
        self.tvs_active.remove('TV57')
        self.tvs_active.remove('TV60')

        self.active_tvinfos = getactive_tvinfos(self.tvs_active)

        initt = time.time()
        self.set_rootdf(df)
        
        self.get_ancinfo(pilimage, t_anc, camid_anc)

        forward, backward, release_forward, release_backward, anchor_cars = self.get_disbase()
        print('\n\n锚点路段dis_avarange\t', str(self.disbase_anc))
        print('backward breakpoint\t', str(self.backward_breakpoint))
        print('backward t\t', str(self.backward_breakpoint_t))
        print('forward breakpoint\t', str(self.forward_breakpoint))
        print('forward t', str(self.forward_breakpoint_t))

        forward_release = self.search_forward(release_forward, self.forward_breakpoint, self.forward_breakpoint_t)
        backward_release = self.search_backward(release_backward, self.backward_breakpoint, self.backward_breakpoint_t)

        self.reinit()
        answer = []
        for car in forward: answer.append(car)
        for car in backward: answer.append(car)
        for car in forward_release: answer.append(car)
        for car in backward_release: answer.append(car)
        
        print(answer)
        return answer
