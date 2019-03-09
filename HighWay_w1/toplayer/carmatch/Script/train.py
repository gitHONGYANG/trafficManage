import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Script.dataloader import ccldataloader,validationDataloader
from Script.model import BaseNetwork
import torch.optim as optim
import torch.nn.functional as F
from operator import itemgetter
from torch.autograd import Variable
import random
from Script.dataloader import ccldataloader
import time
import visdom
from PIL import Image
import cv2
import numpy as np
import torchvision
import os

class ccltrain():
    def __init__(self, initmodelpath,epoch=10):
        self.model = torch.load(initmodelpath)
        self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.2)

        self.transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.ColorJitter(), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #self.optimizer = optim.Adam(self.trimodel.parameters(),lr=0.0005)

    #输入[[1,3,224,224],[1,3,224,224]]的list,组装为[n,3,224,224]进行feature计算
    def ccl_loss(self, positiveimageset, negetiveimageset, margin):
        posbatchsize = len(positiveimageset)
        negbatchsize = len(negetiveimageset)
        posimgbatch = 0
        negimgbatch = 0
        for i in range(posbatchsize):
            if i == 0:
                posimgbatch = positiveimageset[0]
            else:
                posimgbatch = torch.cat([posimgbatch, positiveimageset[i]])
        for i in range(negbatchsize):
            if i == 0:
                negimgbatch = negetiveimageset[0]
            else:
                negimgbatch = torch.cat([negimgbatch, negetiveimageset[i]])

        posfeatures = self.model(posimgbatch)
        negfeatures = self.model(negimgbatch)

        centerfeature = 0
        for i in range(len(posfeatures)):
            if i == 0:
                centerfeature = posfeatures[0]
            else:
                centerfeature = centerfeature + posfeatures[i]
        centerfeature = torch.div(centerfeature, len(posfeatures))

        #归一化
        centerfeature = torch.div(centerfeature, torch.norm(centerfeature, 2))
        centerfeature = torch.unsqueeze(centerfeature, 0)
        #print(centerfeature.shape)

        tempnegfeature = 0
        harddistance = float('Inf')
        hardnegefeature = Variable(torch.zeros(1, 128).cuda())
        for i in range(len(negfeatures)):
            tempnegfeature = negfeatures[i]
            #归一化
            tempnegfeature = torch.div(tempnegfeature, torch.norm(tempnegfeature, 2))
            tempnegfeature = torch.unsqueeze(tempnegfeature, 0)
            #print(tempnegfeature.shape)
            dis = F.pairwise_distance(centerfeature, tempnegfeature, p=2).cpu().data.numpy()[0][0]
            if dis < harddistance:
                harddistance = dis
                hardnegefeature = tempnegfeature

        totalloss = 0
        for i in range(len(posfeatures)):
            temposfeature = posfeatures[i]
            #归一化
            temposfeature = torch.div(temposfeature, torch.norm(temposfeature, 2))
            temposfeature = torch.unsqueeze(temposfeature, 0)
            loss = F.pairwise_distance(temposfeature, centerfeature) + margin - F.pairwise_distance(centerfeature, hardnegefeature)
            loss = torch.clamp(loss, min=0)
            totalloss += loss
        return totalloss

    def train(self, epoch, traindatapath):
        loader = ccldataloader( traindatapath, 4, 0.8, self.model , self.transform)
        print('--------------epoch %d------------ '%epoch)

        trainlist = []
        for posid in loader.getkey():
            for negid in loader.getkey():
                if posid != negid:
                    trainlist.append((posid,negid))

        avgloss = 0
        count = 0
        curtime = time.time()
        while trainlist:

            lr = 0.0002 * (0.9**int(count/5000))
            if lr<0.0001 : lr = 0.0001
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.2)

            #print(trainlist)
            self.optimizer.zero_grad()
            length = len(trainlist)
            popindex = random.randint(0,  length-1)
            currentpair =  trainlist.pop(popindex)
            #print(currentpair)

            posid = currentpair[0]
            negid = currentpair[1]
            loader.setmodel(self.model)

            posimagelist, posimagepathlist, negimagelist, negimagepathlist = loader.getminibatch(posid, negid)
            loss = self.ccl_loss(posimagelist, negimagelist, 0.8)
            #temp = Variable(torch.FloatTensor([[0]]).cuda())
            #temp2 = float(loss[0][0]>temp[0][0])
            #if temp2>0:
            loss.backward()

            avgloss += loss
            self.optimizer.step()
            #print(float(loss))
            #print('\n')

            count += 1
            if count%50 == 0:
                spendtime = time.time() - curtime
                curtime = time.time()
                print('aveloss ',float(avgloss/50), ' %d/%d'  %(count,len(loader.getkey())*(len(loader.getkey())-1)), 'time : %f' %spendtime)
                avgloss = 0
            if count%5000 == 0:
                torch.save(self.model, '../checkpoints/Vgg16_CCL_NORM/epoch' + str(epoch) + '_' + str(count) + '.pt')
                self.validation('../Dataset/similarityValidation-tv35-tv39-num2', self.model)
                print('model save')

    def searchtop5(self, idtv, tvid_centerfeature):
        id = idtv.split('TV')[0]
        tv = idtv.split('TV')[1]

        alltvneedsearch = []
        distancelog = {}
        for key in tvid_centerfeature.keys():
            if key.split('TV')[0] == id and key.split('TV')[1]!= tv:
                tv = key.split('TV')[1]
                alltvneedsearch.append(tv)
                distancelog[tv] = []

        for curTV in alltvneedsearch:
            #搜索curTV中的全部feature
            for key in tvid_centerfeature.keys():
                if key.split('TV')[1] == curTV and key != idtv:
                    anchorfeature = torch.div(tvid_centerfeature[idtv], torch.norm(tvid_centerfeature[idtv], 2))
                    judgefeature = torch.div(tvid_centerfeature[key], torch.norm(tvid_centerfeature[key], 2))
                    distance =  F.pairwise_distance(anchorfeature , judgefeature, p=2).cpu().numpy()[0][0]
                    distancelog[curTV].append([key, distance])

        for searchtv in alltvneedsearch:
            distancelog[searchtv] = sorted(distancelog[searchtv] , key = itemgetter(1))
        for searchtv in alltvneedsearch:
            distancelog[searchtv] = distancelog[searchtv][0:4]

        return distancelog

    def validation(self, validationpath, model):
        #vis = visdom.Visdom(env = 'Validation')
        #vis.close()

        model.cuda()
        valdata = validationDataloader(validationpath)
        valdataset = valdata.getALLFeature(model)
        print('Validation data load success')
        tvid_feature = {}
        tvid_centerfeature = {}
        #valdataset的格式为[(feature, idtv)]
        valnums = len(valdataset)

        for valdata in valdataset:
            if valdata[1] not in tvid_feature.keys():
                tvid_feature[valdata[1]] = []
            tvid_feature[valdata[1]].append(valdata[0])

        tvid_featurekey = list(tvid_feature.keys())
        tvid_featurekey.sort()
        #print(tvid_featurekey)

        for key in tvid_featurekey:
            centerfeature = 0
            for i, feature in enumerate(tvid_feature[key]):
                if i == 0: centerfeature = feature
                else:
                    centerfeature = centerfeature + feature
            centerfeature = torch.div(centerfeature, len(tvid_feature[key]))
            tvid_centerfeature[key] = centerfeature

        totalvalnum = 0
        rightnum = 0
        wrongnum = 0
        log = open('distancelog.txt', 'w')

        havesearchedlist = []
        for valdata in valdataset:
            if valdata[1].split('TV')[1] == '35' and valdata[1] not in havesearchedlist:
                #print(valdata[1])
                havesearchedlist.append(valdata[1])
                anchorid = valdata[1].split('TV')[0]
                anchorpath = valdata[2]
                answer = self.searchtop5(valdata[1], tvid_centerfeature)

                line = valdata[1] + '  ' + str(answer) +'\n'
                log.writelines(line)

                for searchtv in answer.keys():
                    totalvalnum += 1
                    if anchorid == answer[searchtv][0][0].split('TV')[0]:
                        rightnum += 1
                    else:
                        vistransform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
                        anchorimg = vistransform(Image.open(anchorpath)).view(1,3,224,224)

                        print(answer[searchtv][0][0])
                        count = 0
                        for imagename in os.listdir(validationpath):
                            if imagename.split('_')[0] + imagename.split('_')[4] == answer[searchtv][0][0]:
                                negimg = vistransform(Image.open(validationpath + '/'+ imagename)).view(1,3,224,224)
                                imagespair = torch.cat([anchorimg, negimg])
                                title = anchorid + 'TV35---' + imagename.split('_')[0] + imagename.split('_')[4]
                                #if count == 0: vis.images(imagespair, opts={'title': title})
                                count += 1
                        #unmatchimg = transforms.ToTensor()(Image.open(answer[searchtv]))
                #print(valdata[1], ' valdone\n')
        print('totalnum ', str(totalvalnum))
        print('rightnum ', str(rightnum))
        print('wrongnum ', str(wrongnum))
        print('Acc  ', str(rightnum/totalvalnum), '\n')
        log.close()

        #print(answer)

def test():

    image = Image.open('../FunctionTest/1.jpg')
    image = transforms.ToTensor()(image)
    vis = visdom.Visdom(env= u'test1')
    vis.close()
    vis.text('hello')
    vis.image(image)

if __name__ == '__main__':
    t = ccltrain('../checkpoints/Vgg16_CCL_NORM/epoch1_70000.pt')
    t.validation('../Dataset/similarityValidation-tv35-tv39-num2', t.model)
    for i in range(2,5):
        t.train(i, '../Dataset/similarityTrain-tv35-tv39')