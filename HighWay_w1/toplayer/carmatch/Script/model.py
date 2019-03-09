import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.autograd import Variable


class BaseNetwork(nn.Module):
    def __init__(self, modelname):
        super(BaseNetwork, self).__init__()
        self.modelname = modelname

        if modelname == 'Vgg16':
            self.CNN = models.vgg16(pretrained=True).features
            self.FC1 = nn.Linear(7*7*512, 2048)
            self.FC2 = nn.Linear(2048, 128)
        elif modelname == 'Resnet50':
            print(models.resnet50())
            self.CNN = nn.Sequential( *list(models.resnet50(pretrained=True).children())[:-1])
            self.FC = nn.Linear(2048, 128)
        else:
            raise ('Please select model')

    def forward(self, x):
        if self.modelname == 'Vgg16':
            output = self.CNN(x)
            output = output.view(output.size()[0], -1)
            output = self.FC1(output)
            output = F.relu(output)
            output = self.FC2(output)
        elif self.modelname == 'Resnet50':
            output = self.CNN(x)
            output = output.view(output.size()[0], -1)
            output = F.relu(output)
            output = self.FC(output)
        else:
            raise ('Please select model')

        return output

if __name__ == '__main__':
    modelname = 'Vgg16'
    embedding = BaseNetwork(modelname)
    print(embedding)
    if not os.path.exists('../checkpoints/' + modelname):
        os.mkdir('../checkpoints/' + modelname)
        print('mkdir')

    savename = modelname + '_CCL'
    torch.save(embedding, '../checkpoints/' + savename + '/epoch0.pt')

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = Image.open('../FunctionTest/1.jpg')
    img = Variable(transform(img)).view(1, 3, 224, 224)
    feature = embedding(img)
    print(feature)
