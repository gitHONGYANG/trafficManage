from core.service import ImageProvider
import cv2
import time
import pymongo
from pymongo import MongoClient
import base64
from PIL import Image
from io import BytesIO
import numpy

class VideoProvider(ImageProvider):
    def __init__(self, id, cameraid):
        self.id = id
        self.imgtb = MongoClient('localhost:27017')['highway']['temp_img']
        self.cameraid = cameraid
        self.frame_count = -1        

    def impulse(self):
        img, t = self.get_newimg()
        return {'id': self.id, 'img': img, 'time': t}

    def get_newimg(self):
        info = self.imgtb.find_one({'cameraid' : self.cameraid},{'tempimg' : 1, 'time':1, 'count':1})
        img = info['tempimg']
        t = info['time']
        frame_count = info['count']
        
        if frame_count!= self.frame_count:
            self.frame_count = frame_count

            img = Image.open(BytesIO(base64.b64decode(img)))
            self.img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

        return self.img, t

    def ok(self):
        self.status = True
        #self.status, self.frame = self.cap.read()
        if not self.status: print('video end')
        return self.status
