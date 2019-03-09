from Script.model import BaseNetwork
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import random
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import time
from Script.utils import printlist, printtable

class ccldataloader():
    def __init__(self, rawimagepath, minibatchsize, margin, model, transform):
        self.minibatchsize = minibatchsize
        self.margin = margin
        self.model = model

        self.imagepathlist =[ rawimagepath + '/' + imagename for imagename in os.listdir(rawimagepath)]
        self.imageidlist = [image.split('_')[0] for image in os.listdir(rawimagepath)]

        self.imageidpathmap = {}
        self.imageidtvpathmap = {}
        self.getimageidpathmap()

        self.transform = transform

        #printtable(self.imageidpathmap, 'imageidpathmap')
    def getimageidpathmap(self):
        #建立{ ID- [path1, path2] }的table
        caridnum = len(self.imageidlist)
        for index in range(caridnum):
            if self.imageidlist[index] not in self.imageidpathmap.keys():
                self.imageidpathmap[self.imageidlist[index]] = []
            self.imageidpathmap[self.imageidlist[index]].append(self.imagepathlist[index])

        for id in self.imageidpathmap.keys():
            fullimagelist = self.imageidpathmap[id]
            for image in fullimagelist:
                tv = image.split('/')[-1].split('_')[4]
                tvID = id + '_' + tv
                if tvID not in self.imageidtvpathmap:
                    self.imageidtvpathmap[tvID] = []
                self.imageidtvpathmap[tvID].append(image)


    def printmap(self, id):
        for path in self.imageidpathmap[id]:
            print(path)

        for key in self.imageidtvpathmap.keys():
            if key.split('_')[0]==id:
                print(self.imageidtvpathmap[key])

    def getkey(self):
        return list(self.imageidpathmap.keys())

    def setmodel(self, model):
        self.model = model

    def getimagefromID(self, id, classnum , everytvnum = 2):
        imagelist = []
        totalnum = classnum * everytvnum

        for key in self.imageidtvpathmap.keys():
            if str(id) == key.split('_')[0]:
                if len(self.imageidtvpathmap[key]) >= everytvnum:
                    tempimagelist = random.sample(self.imageidtvpathmap[key], everytvnum)
                    for image in tempimagelist:
                        imagelist.append(image)
                else:
                    for image in self.imageidtvpathmap[key]:
                        imagelist.append(image)

        for i in range(totalnum - len(imagelist)):
            imagelist.append(random.choice(self.imageidpathmap[id]))

        #printlist(imagelist, 'imagelist')
        return imagelist

    def getimagefeature(self, imagepathlist):
        imagefeatures = []
        imagenum = len(imagepathlist)
        centerfeature = Variable(torch.zeros(1, 128).cuda())
        for imagepath in imagepathlist:
            img = Image.open(imagepath)
            img = Variable(self.transform(img).cuda()).view(1, 3, 224, 224)
            feature = self.model(img)
            feature = torch.div(feature, torch.norm(feature, 2))
            imagefeatures.append(feature)

            centerfeature += feature
        centerfeature = torch.div(centerfeature, imagenum)

        return imagefeatures, centerfeature
        #printlist(imagefeature, 'imagefeature')

    def getminibatch(self, positiveid, negetiveid, everytvnum = 2):
        posimagepathlist = self.getimagefromID(positiveid, 5, everytvnum)
        posimagelist = []
        for imagepath in posimagepathlist:
            img = Image.open(imagepath)
            img = Variable(self.transform(img).cuda()).view(1, 3, 224, 224)
            posimagelist.append(img)

        negimagepathlist = self.getimagefromID(negetiveid, 5, everytvnum)
        negimagelist = []
        for imagepath in negimagepathlist:
            img = Image.open(imagepath)
            img = Variable(self.transform(img).cuda()).view(1, 3, 224, 224)
            negimagelist.append(img)

        return posimagelist, posimagepathlist, negimagelist, negimagepathlist

class validationDataloader():
    def __init__(self, rawimagepath):
        self.rootpath = rawimagepath
        self.imagepathlist = [rawimagepath + '/' + imagename for imagename in os.listdir(rawimagepath)]
        self.imageidlist = [image.split('_')[0] for image in os.listdir(rawimagepath)]

        self.imageidpathmap = {}
        self.imageidtvpathmap = {}
        self.getimageidpathmap()

    def getimageidpathmap(self):
        #建立{ ID- [path1, path2] }的table
        caridnum = len(self.imageidlist)
        for index in range(caridnum):
            if self.imageidlist[index] not in self.imageidpathmap.keys():
                self.imageidpathmap[self.imageidlist[index]] = []
            self.imageidpathmap[self.imageidlist[index]].append(self.imagepathlist[index])

        for id in self.imageidpathmap.keys():
            fullimagelist = self.imageidpathmap[id]
            for image in fullimagelist:
                tv = image.split('/')[-1].split('_')[4]
                tvID = id + '_' + tv
                if tvID not in self.imageidtvpathmap:
                    self.imageidtvpathmap[tvID] = []
                self.imageidtvpathmap[tvID].append(image)

    def getALLFeature(self, model):
        self.getimageidpathmap()
        imageinfolist = []
        model.cuda()
        valtransform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        for imagepath in self.imagepathlist:
            image = Image.open(imagepath)
            image = Variable(valtransform(image).cuda()).view(1, 3, 224, 224)
            idtv = imagepath.split('/')[-1].split('_')[0] + imagepath.split('/')[-1].split('_')[4]
            feature = model(image)
            #print(feature)
            imageinfolist.append((feature.data,idtv,imagepath))
        return imageinfolist

def test():
    '''
    loader = tripletBatchDataLoader('../checkpoints/embedding_epoch0.pt', '../Dataset/Cardataset_K898AndK900_243car',4, 1.0)
    init = time.time()
    for i in range(100):
        print(i)
        anchorbatch, positivebatch, negetivebatch = loader.getminibatch()
    print(time.time()-init)
    '''
    val = validationDataloader('../Dataset/val_Cardataset_K898AndK900')
    list = val.getAll()
    print(list)

if __name__ == '__main__':

    model = torch.load('../checkpoints/Vgg16_CCL/epoch0.pt')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #load = ccldataloader( '../Dataset/similarityTrain-tv35-tv39', 4, 1.0, model , transform)
    #load.getimagefromID('1', 5, everytvnum=3)
    load = validationDataloader( '../Dataset/similarityValidation-tv35-tv39')
    load.getimageidpathmap()

    features = load.getALLFeature()
    '''
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = torch.load('../checkpoints/Resnet50/epoch0.pt')
    model.cuda()
    l = ccldataloader( '../Dataset/train_Cardataset_K898AndK900', 4, 1.0, model , transform)
    #list = l.getimagefromID('1', 5)
    #l.getimagefeature(list)
    posimagelist, posimagepathlist, negimagelist, negimagepathlist = l.getminibatch('1', '3')
    print(posimagelist[0].shape)
    '''