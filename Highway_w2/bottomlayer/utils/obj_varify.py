#本脚本主要做的是异物识别
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from torch.autograd import Variable
import time
import cv2

class Net (nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.modle = models.resnet34(pretrained =True)
        num_ftrs = self.modle.fc.in_features
        self.modle.fc = nn.Linear(num_ftrs, 32)
        self.FC1 = nn.Linear(32, 3)

    def forward(self, x):
        output = self.modle(x)
        output = F.sigmoid(output)
        output = self.FC1(output)
        return output

'''
validation(img)
return Ture:异物
return False:误报
'''

class ObjValidation():
    def __init__(self, modelpath):
        self.model = Net()
        #print(self.model)
        self.model.load_state_dict(torch.load(modelpath))
        self.model.cuda()
        self.transform = transforms.Compose(  #预处理
            [transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    def varify_pil(self, pilimage):
        image = Variable(self.transform(pilimage).unsqueeze(0).cuda())  #batch = 1线上模式
        output = self.model(image)
        #模型的outchannel的数目为3
        output = output.cpu().data.numpy()


        print(output)
        answer = False
        if output[0][2]>output[0][0] and output[0][2]>output[0][1]:
            answer = True
        elif output[0][0]>1 + output[0][1] and output[0][0]>1 + output[0][2]:
            #print('car')
            #pilimage.save('car.jpg')
            answer = False
        else:
            answer = False
        return answer

    def varify_cv(self, cvimage):
        pilimage = Image.fromarray(cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB))
        answer = self.varify_pil(pilimage)
        return answer

if __name__ == '__main__':
    modelpath = '/media/assests/checkpoints/obj_resnet_29.pt'
    objvalprocess = ObjValidation(modelpath)
    initt = time.time()
    trueimage = cv2.imread('obj.jpg')
    falseimage = cv2.imread('false.jpg')
    carimage = cv2.imread('car.jpg')
    
    ifobj = objvalprocess.varify_cv(carimage)
    print(ifobj)
    ifobj = objvalprocess.varify_cv(trueimage)
    print(ifobj)
    ifobj = objvalprocess.varify_cv(falseimage)
    print(ifobj)

    '''
    for i in range(100):
        ifobj = objvalprocess.varify_cv(trueimage)
        print(ifobj)
        ifobj = objvalprocess.varify_cv(falseimage)
        print(ifobj)
    print(time.time() - initt)
    '''
