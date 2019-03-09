import pandas as pd
import os
import numpy as np
import cv2
import time
import datetime
import shutil
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from ipdb import set_trace
import pymongo
import json
#地理对应表
forwardtvdistance = {'TV36':2000, 'TV37':1970, 'TV38':1030, 'TV39':1000, 'TV40':1020, 'TV41':980, 'TV42':2010, 'TV43':1010, 'TV44':980,
              'TV45':990, 'TV46':1010, 'TV47':1000, 'TV48':1000, 'TV49':2010, 'TV50':1990, 'TV51':1470, 'TV52':950, 'TV53':2000, 'TV54':3000,
              'TV55':980, 'TV56':1000, 'TV57':1020, 'TV58':1970, 'TV59':1980, 'TV60':1440, 'TV61':1430, 'TV62':890, 'TV63':1380}

backwardtvdistance = {'TV35':2000, 'TV36':1970, 'TV37':1030, 'TV38':1000, 'TV39':1020, 'TV40':980, 'TV41':2010, 'TV42':1010, 'TV43':980,
              'TV44':990, 'TV45':1010, 'TV46':1000, 'TV47':1000, 'TV48':2010, 'TV49':1990, 'TV50':1470, 'TV51':950, 'TV52':2000, 'TV53':3000,
              'TV54':980, 'TV55':1000, 'TV56':1020, 'TV57':1970, 'TV58':1980, 'TV59':1440, 'TV60':1430, 'TV61':890, 'TV62':1380}

forwardtvmap ={}
backwardtvmap = {}
for i in range(35,63):
    key = 'TV%d' %i
    value = 'TV%d' %(i+1)
    forwardtvmap[key] = value

for i in range(36,64):
    key = 'TV%d' %i
    value = 'TV%d' %(i-1)
    backwardtvmap[key] = value

modelpath = '../../model/CCL_epoch10_305000.pt'

class similarity():
    def __init__(self):
        self.model = torch.load(modelpath).cuda()
        self.transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #以list形式返回feature
    def getembeddingfrompil(self, pilimage):
        starttime = time.time()
        tensorimage = Variable(self.transform(pilimage)).view(1, 3, 224, 224).cuda()
        feature = self.model(tensorimage)
        feature = list(feature.cpu().data.numpy().flatten()*1000)
        feature = [int(f) for f in feature]
        #print('cal feature')
        #print(time.time()-starttime)
        return feature

class carmatchutils():
    #读取CSV文件,获取图像与车辆全部信息
    def __init__(self):
        #self.carinfodf = pd.read_csv(carinfocsvpath)

        #self.tb = pymongo.MongoClient(host = 'localhost', port=27017)['highway'][tbname]
        #self.carinfodf = pd.DataFrame(list(self.tb.find()))

        #self.carinfodf = df
        #self.carinfodf = pd.read_csv('highway_carinfo.csv')
        self.similarity = similarity()
        #print(self.carinfodf.info())

    def setcarinfodf(self,df):
        self.carinfodf = df

    #提取锚点PIL图像feature(已乘1000统一度量)
    def getanchorcarinfo(self, pilimage, time, tvid):
        anchorcarfeature = self.similarity.getembeddingfrompil(pilimage)
        return anchorcarfeature , time, tvid

    #通过锚点PIL图像feature,tvid,time 查找所在库中的车辆图片
    def getinitialanchorcar(self, anchorcarfeature, time, tvid):
        searchdf = self.selectcarinfobytvandtime(self.carinfodf, tvid, time-10, time+10)
        #print(searchdf)
        return searchdf

    #选择CSV中对应TV号的信息
    def selectcarinfobytv(self, tvid):
        caroftvdf = self.carinfodf[self.carinfodf['cameraid'] == tvid]
        return caroftvdf

    #在输入数据库中选择对应时间段
    def selecttimefromdf(self, df, starttime, endtime):
        carintimedf = df[(df['time']>starttime) & (df['time']<endtime)]
        return carintimedf

    #默认输入self.carinfodf
    def selectcarinfobytvandtime(self, df, tvid, starttime, endtime):
        carsdf = df[(df['cameraid'] == tvid) & (df['time']>starttime) & (df['time']<endtime)]
        return carsdf


    #在给定库中搜索相似度最高的三辆车,返回[[distance, carid, feture]]
    #anchorfeature为从pil中选择得到图像通过网络获得的特征，为list
    def searchsamecar(self, anchorfeature, searchcardf):
        searchcardf = searchcardf.copy()

        searchcardf['dist'] = searchcardf['feature'].apply(
                lambda x:np.linalg.norm(np.array(x)-anchorfeature)
                )
        searchcardf = searchcardf.sort_values(by='dist')
        v = searchcardf[['dist', 'carid', 'feature']].values
        return v[:3]

    #实际使用需要继续修改，这里固定了查询的数据库
    #查询carid对应的车辆出现的时间、地点、位置、图像路径
    def getcarinfosfromid(self, carid):
        inittime = time.time()
        starttime = time.time()

        #carid = str(carid)
        msk = self.carinfodf['carid']==carid
        starttime = time.time()
        idx = self.carinfodf.index[msk][0]
        starttime = time.time()
        rst = self.carinfodf.loc[idx, ['imagepath', 'time', 'location', 'cameraid']]
        return rst


    def searchthreeclosecar(self, anchorfeature, cardf):
        samecarlist = self.searchsamecar(anchorfeature, cardf)

        samecarinfos = []

        for samecar in samecarlist:
            imagepath, t, location, cameraid = self.getcarinfosfromid(samecar[1])
            distancefromanchor = samecar[0]
            originalfeature = samecar[2]

            samecarinfos.append([t, imagepath, location, cameraid, distancefromanchor, originalfeature])

        return samecarinfos

    #
    def roughsearch(self,anchorfeature, inittime, initcamera):
        startt = time.time()

        carcamera = initcamera
        caroccurtime = inittime
        #print('锚点车辆所在摄像头 : ', carcamera)
        #print('锚点车辆出现时间   : ', caroccurtime)

        forwardcamera = carcamera
        backwardcamera = carcamera
        forwardtime = caroccurtime
        backwardtime = caroccurtime

        roughsearchsamecarlist = []

        searchdf = self.selectcarinfobytvandtime(self.carinfodf, initcamera, inittime-10, inittime+10)

        #startt = time.time()
        samecars = self.searchthreeclosecar(anchorfeature, searchdf)
        #保存[时间，图片路径，定位框，与锚点距离，tv号]
        for samecar in samecars:
            roughsearchsamecarlist.append(samecar[0:5])

        while forwardcamera:
            forwardcamera = forwardtvmap.get(forwardcamera, False)
            if not forwardcamera:
                break

            if forwardcamera == 'TV52':
                forwardsearchstarttime = forwardtime + forwardtvdistance[forwardcamera]*3.6/120
                forwardsearchendtime = forwardtime + 240
            else:
                forwardsearchstarttime = forwardtime + forwardtvdistance[forwardcamera]*3.6/120
                forwardsearchendtime = forwardtime + forwardtvdistance[forwardcamera]*3.6/40

            searchdf = self.selectcarinfobytvandtime(self.carinfodf, forwardcamera, forwardsearchstarttime, forwardsearchendtime)
            samecarinfolist = self.searchthreeclosecar(anchorfeature, searchdf)
            if not samecarinfolist:
                break

            forwardtime = 0
            for i in range(len(samecarinfolist)):
                forwardtime += samecarinfolist[i][0]

            forwardtime = forwardtime/len(samecarinfolist)

            for carinfo in samecarinfolist:
                roughsearchsamecarlist.append(carinfo[0:5])

        while backwardcamera:
            backwardcamera = backwardtvmap.get(backwardcamera)
            if not backwardcamera:
                break

            if backwardcamera == 'TV51':
                backwardsearchstarttime = backwardtime - 240
                backwardsearchendtime = backwardtime - backwardtvdistance[backwardcamera]*3.6/120
            else:
                backwardsearchstarttime = backwardtime - backwardtvdistance[backwardcamera]*3.6/40
                backwardsearchendtime = backwardtime - backwardtvdistance[backwardcamera]*3.6/120

            searchdf = self.selectcarinfobytvandtime(self.carinfodf, backwardcamera, backwardsearchstarttime, backwardsearchendtime)
            samecarinfolist = self.searchthreeclosecar(anchorfeature, searchdf)
            if not samecarinfolist:
                break
            backwardtime = 0
            for i in range(len(samecarinfolist)):
                backwardtime += samecarinfolist[i][0]
            backwardtime = backwardtime/len(samecarinfolist)

            for carinfo in samecarinfolist:
                roughsearchsamecarlist.append(carinfo[0:5])

        roughsearchsamecarlist.sort()
        #endt = time.time()
        #print('roughsearch  ', str(endt - startt))

        return roughsearchsamecarlist

    #输入的samecarlist可能包含多段道路相似车辆，该函数返回各路段的距离均值
    def getavadistance(self, samecarlist):
        tv35totv43avadistance = 0
        tv44totv50avadistance = 0
        tv51totv55avadistance = 0
        tv57totv59avadistance = 0
        tv61totv63avadistance = 0

        tv35totv43count = 0
        tv44totv50count = 0
        tv51totv55count = 0
        tv57totv59count = 0
        tv61totv63count = 0

        tv35totv43 = ['TV%d' %i for i in range(35, 44)]
        tv44totv50 = ['TV%d' %i for i in range(44, 51)]
        tv51totv55 = ['TV%d' %i for i in range(51, 56)]
        tv57totv59 = ['TV%d' %i for i in range(57, 60)]
        tv61totv63 = ['TV%d' %i for i in range(61, 64)]

        for samecar in samecarlist:
            if samecar[3] in tv35totv43:
                tv35totv43count += 1
                tv35totv43avadistance += samecar[4]
            elif samecar[3] in tv44totv50:
                tv44totv50count += 1
                tv44totv50avadistance += samecar[4]
            elif samecar[3] in tv51totv55:
                tv51totv55count += 1
                tv51totv55avadistance += samecar[4]
            elif samecar[3] == 'TV56':
                tv57totv59count += 1
                tv57totv59avadistance += samecar[4]
            elif samecar[3] in tv57totv59:
                tv57totv59count += 1
                tv57totv59avadistance += samecar[4]
            elif samecar[3] == 'TV60':
                tv61totv63count += 1
                tv61totv63avadistance += samecar[4]

            elif samecar[3] in tv61totv63:
                tv61totv63count += 1
                tv61totv63avadistance += samecar[4]
            else:
                print(samecar)
                print('Error: funciton getavadistace')

        if tv35totv43count==0:
            tv35totv43avadistance = float('Inf')
        else:
            tv35totv43avadistance = tv35totv43avadistance/tv35totv43count
        if tv44totv50count==0:
            tv44totv50avadistance = float('Inf')
        else:
            tv44totv50avadistance = tv44totv50avadistance/tv44totv50count
        if tv51totv55count==0:
            tv51totv55avadistance = float('Inf')
        else:
            tv51totv55avadistance = tv51totv55avadistance/tv51totv55count
        if tv57totv59count==0:
            tv57totv59avadistance = float('Inf')
        else:
            tv57totv59avadistance = tv57totv59avadistance/tv57totv59count
        if tv61totv63count==0:
            tv61totv63avadistance = float('Inf')
        else:
            tv61totv63avadistance = tv61totv63avadistance/tv61totv63count

        return tv35totv43avadistance, tv44totv50avadistance, tv51totv55avadistance, tv57totv59avadistance, tv61totv63avadistance

    def distancefilterforsamecars(self, samecarlist, anchorfeature, anchorcameraid):
        tv35totv43 = ['TV%d' %i for i in range(35, 44)]
        tv44totv50 = ['TV%d' %i for i in range(44, 51)]
        tv51totv56 = ['TV%d' %i for i in range(51, 57)]
        tv57totv59 = ['TV%d' %i for i in range(57, 60)]
        tv61totv63 = ['TV%d' %i for i in range(61, 64)]

        tv35totv43avadistance, tv44totv50avadistance, tv51totv55avadistance, tv57totv59avadistance, tv61totv63avadistance = self.getavadistance(samecarlist)
        '''
        print(tv35totv43avadistance)
        print(tv44totv50avadistance)
        print(tv51totv55avadistance)
        print(tv57totv59avadistance)
        print(tv61totv63avadistance)
        '''
        finalsegments = set(['TV%d' %i for i in range(35,64)])
        if anchorcameraid in tv35totv43:
            anchorsegmentdistance =tv35totv43avadistance
            if tv35totv43avadistance > anchorsegmentdistance*1.5:
                pass
            if tv44totv50avadistance > anchorsegmentdistance*1.5:
                finalsegments = finalsegments - set(['TV%d' %i for i in range(44, 64)])
            if tv51totv55avadistance > anchorsegmentdistance*1.5:
                finalsegments = finalsegments - set(['TV%d' %i for i in range(51, 64)])
            if tv57totv59avadistance > anchorsegmentdistance*1.5:
                finalsegments = finalsegments - set(['TV%d' %i for i in range(56, 64)])
            if tv61totv63avadistance > anchorsegmentdistance*1.5:
                finalsegments = finalsegments - set(['TV%d' %i for i in range(60, 64)])

        elif anchorcameraid in tv44totv50:
            anchorsegmentdistance =tv44totv50avadistance
            if tv35totv43avadistance > anchorsegmentdistance*1.5:
                finalsegments =finalsegments - set(['TV%d' %i for i in range(35, 44)])
            if tv44totv50avadistance > anchorsegmentdistance*1.5:
                pass
            if tv51totv55avadistance > anchorsegmentdistance*1.5:
                finalsegments = finalsegments - set(['TV%d' %i for i in range(51, 64)])
            if tv57totv59avadistance > anchorsegmentdistance*1.5:
                finalsegments = finalsegments - set(['TV%d' %i for i in range(56, 64)])
            if tv61totv63avadistance > anchorsegmentdistance*1.5:
                finalsegments = finalsegments - set(['TV%d' %i for i in range(60, 64)])

        elif anchorcameraid in tv51totv56:
            anchorsegmentdistance =tv51totv55avadistance
            if tv35totv43avadistance > anchorsegmentdistance*1.5:
                finalsegments =finalsegments - set(['TV%d' %i for i in range(35, 44)])
            if tv44totv50avadistance > anchorsegmentdistance*1.5:
                finalsegments =finalsegments - set(['TV%d' %i for i in range(35, 51)])
            if tv51totv55avadistance > anchorsegmentdistance*1.5:
                pass
            if tv57totv59avadistance > anchorsegmentdistance*1.5:
                finalsegments = finalsegments - set(['TV%d' %i for i in range(56, 64)])
            if tv61totv63avadistance > anchorsegmentdistance*1.5:
                finalsegments = finalsegments - set(['TV%d' %i for i in range(60, 64)])

        elif anchorcameraid in tv57totv59:
            anchorsegmentdistance = tv57totv59avadistance
            if tv35totv43avadistance > anchorsegmentdistance*1.5:
                finalsegments =finalsegments - set(['TV%d' %i for i in range(35, 44)])
            if tv44totv50avadistance > anchorsegmentdistance*1.5:
                finalsegments =finalsegments - set(['TV%d' %i for i in range(35, 51)])
            if tv51totv55avadistance > anchorsegmentdistance*1.5:
                finalsegments =finalsegments - set(['TV%d' %i for i in range(35, 57)])
            if tv57totv59avadistance > anchorsegmentdistance*1.5:
                pass
            if tv61totv63avadistance > anchorsegmentdistance*1.5:
                finalsegments = finalsegments - set(['TV%d' %i for i in range(60, 64)])

        elif anchorcameraid in tv61totv63:
            anchorsegmentdistance =tv61totv63avadistance
            if tv35totv43avadistance > anchorsegmentdistance*1.5:
                finalsegments =finalsegments - set(['TV%d' %i for i in range(35, 44)])
            if tv44totv50avadistance > anchorsegmentdistance*1.5:
                finalsegments =finalsegments - set(['TV%d' %i for i in range(35, 51)])
            if tv51totv55avadistance > anchorsegmentdistance*1.5:
                finalsegments =finalsegments - set(['TV%d' %i for i in range(35, 57)])
            if tv57totv59avadistance > anchorsegmentdistance*1.5:
                finalsegments = finalsegments - set(['TV%d' %i for i in range(35, 61)])
            if tv61totv63avadistance > anchorsegmentdistance*1.5:
                pass

        finalsegments = list(finalsegments)
        #print(finalsegments)
        finalsamecarlist = []
        for samecar in samecarlist:
            if samecar[3] in finalsegments:
                finalsamecarlist.append(samecar)

        return finalsamecarlist, finalsegments


    def searchforservicezone(self, samecarlist, samecarsegment, anchorfeature):

        if ('TV51' in samecarsegment) and ('TV50' not in samecarsegment):
            print('\nservice zone searching.\n')

            avadistance51to55 = 0
            discount = 0
            searchstarttime = 0
            for samecar in samecarlist:
                if samecar[3]  in ['TV%d' %i for i in range(51, 56)]:
                    avadistance51to55 += samecar[4]
                    discount += 1
                if samecar[3] == 'TV51':
                    searchstarttime = samecar[0]
            avadistance51to55 = avadistance51to55/discount

            #search
            searchcount = 0
            while True:
                backwardcamera = 'TV50'

                starttime = searchstarttime - 1800*(searchcount +1)
                endtime = searchstarttime -1800*(searchcount)
                searchdf = self.selectcarinfobytvandtime(self.carinfodf, 'TV50', starttime, endtime)

                samecarinfolist = self.searchthreeclosecar(anchorfeature, searchdf)

                if not samecarinfolist:
                    print('search end, not found')
                    return None, None

                avadistance = 0
                for i in range(len(samecarinfolist)):
                    avadistance += samecarinfolist[0][4]
                avadistance = avadistance/len(samecarinfolist)
                backwardtime = samecarinfolist[0][0]

                if avadistance < 1.2*avadistance51to55:
                    print('search end, found')
                    return backwardtime, 'backward'

                searchcount += 1

        elif ('TV50' in samecarsegment) and ('TV51' not in samecarsegment):
            print('searching.')
            #print('searching..')
            #print('searching...')

            avadistance44to50 = 0
            discount = 0
            searchstarttime = 0
            for samecar in samecarlist:
                if samecar[3]  in ['TV%d' %i for i in range(44, 51)]:
                    avadistance44to50 += samecar[4]
                    discount += 1
                if samecar[3] == 'TV50':
                    searchstarttime = samecar[0]
            avadistance44to50 = avadistance44to50/discount

            #search
            searchcount = 0
            while True:
                backwardcamera = 'TV51'

                starttime = searchstarttime + 1800*(searchcount)
                endtime = searchstarttime + 1800*(searchcount+1)
                searchdf = self.selectcarinfobytvandtime(self.carinfodf, 'TV51', starttime, endtime)
                samecarinfolist = self.searchthreeclosecar(anchorfeature, searchdf)
                if not samecarinfolist:
                    print('search end, not found')
                    return None, None

                avadistance = 0
                for i in range(len(samecarinfolist)):
                    avadistance += samecarinfolist[0][4]
                avadistance = avadistance/len(samecarinfolist)
                backwardtime = samecarinfolist[0][0]

                if avadistance < 1.2*avadistance44to50:
                    print('search end, found')
                    return backwardtime, 'forward'

                searchcount += 1
        else:
            return None, None

    def getfinalsequence(self, anchorfeature, direction, matchcartime):
        finalsamecarlist = []
        if direction == None:
            pass
        elif direction == 'forward':
            forwardcamera = 'TV50'
            forwardtime = matchcartime

            while forwardcamera:
                forwardcamera = forwardtvmap.get(forwardcamera, False)
                if not forwardcamera:
                    break

                if forwardcamera == 'TV51':
                    forwardsearchstarttime = forwardtime -10
                    forwardsearchendtime = forwardtime + 10
                elif forwardcamera == 'TV52':
                    forwardsearchstarttime = forwardtime + forwardtvdistance[forwardcamera]*3.6/120
                    forwardsearchendtime = forwardtime + forwardtvdistance[forwardcamera]*3.6/40 + 60
                else:
                    forwardsearchstarttime = forwardtime + forwardtvdistance[forwardcamera]*(3.6/120)
                    forwardsearchendtime = forwardtime + forwardtvdistance[forwardcamera]*(3.6/40)

                searchdf = self.selectcarinfobytvandtime(self.carinfodf, forwardcamera, forwardsearchstarttime, forwardsearchendtime)
                samecarinfolist = self.searchthreeclosecar(anchorfeature, searchdf)


                print('----forward----')
                print(samecarinfolist[0][:5])
                if not samecarinfolist:
                    break


                forwardtime = 0
                for i in range(len(samecarinfolist)):
                    forwardtime += samecarinfolist[i][0]
                    print('\t', str(samecarinfolist[i][0]))
                forwardtime = forwardtime/len(samecarinfolist)

                print(forwardtime)

                for carinfo in samecarinfolist:
                    finalsamecarlist.append(carinfo[0:5])

        elif direction == 'backward':
            backwardcamera = 'TV51'
            backwardtime = matchcartime
            while backwardcamera:
                backwardcamera = backwardtvmap.get(backwardcamera)
                if not backwardcamera:
                    break

                if backwardcamera == 'TV50':
                    backwardsearchstarttime = backwardtime - 10
                    backwardsearchendtime = backwardtime + 10
                else:
                    backwardsearchstarttime = backwardtime - backwardtvdistance[backwardcamera]*3.6/40
                    backwardsearchendtime = backwardtime - backwardtvdistance[backwardcamera]*3.6/120

                searchdf = self.selectcarinfobytvandtime(self.carinfodf, backwardcamera, backwardsearchstarttime, backwardsearchendtime)
                samecarinfolist = self.searchthreeclosecar(anchorfeature, searchdf)
                if not samecarinfolist:
                    break
                backwardtime = 0
                for i in range(len(samecarinfolist)):
                    backwardtime += samecarinfolist[i][0]
                backwardtime = backwardtime/len(samecarinfolist)

                for carinfo in samecarinfolist:
                    finalsamecarlist.append(carinfo[0:5])

        finalsamecarlist.sort()
        return finalsamecarlist

    def searchcar(self, pilimage, inittime, initcameraid, df):
        self.carinfodf = df
        feature, inittime, initcameraid = self.getanchorcarinfo(pilimage, inittime, initcameraid)
        roughsearchlist = self.roughsearch(feature, inittime, initcameraid)
        modifylist, modifysegments = self.distancefilterforsamecars(roughsearchlist, feature, initcameraid)
        matchtime, direction = self.searchforservicezone(modifylist, modifysegments, feature)

        finalsamecarlist = self.getfinalsequence(feature, direction, matchtime) + modifylist

        answer, answersegments = self.distancefilterforsamecars(finalsamecarlist, feature, initcameraid)

        return answer

if __name__ == "__main__":
    carmatch = carmatchutils('carmatchinfo','./model/CCL_epoch10_305000.pt')
    testimage = Image.open('tv47_656.240.jpg')

    startt = time.time()
    initt = datetime.datetime(2018, 7,14,12,0,0) + datetime.timedelta(seconds=656)
    t1 = int(time.mktime(initt.timetuple()))
    t = float('{}{:06}'.format(t1, initt.microsecond))/1000000
    cameraid = 'TV47'

    answer = carmatch.searchcar(testimage, t, cameraid)

    for info in answer:
        print(info[1])
    '''
    feature, inittime, initcameraid = carmatch.getanchorcarinfo(testimage, t2, 'TV47')

    roughlist = carmatch.roughsearch(feature, t2, 'TV47')

    modifylist, modifysegments = carmatch.distancefilterforsamecars(roughlist, feature, 'TV47')
    matchtime, direction = carmatch.searchforservicezone(modifylist, modifysegments, feature)
    finalsamecarlist = carmatch.getfinalsequence(feature, direction, matchtime)
    print('total ', str(time.time()-startt))
    '''
