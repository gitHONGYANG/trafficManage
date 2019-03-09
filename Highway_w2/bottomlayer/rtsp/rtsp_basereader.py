import cv2
from pymongo import MongoClient
import time
import sys
import os
import datetime
import base64

url = 'mongodb://192.168.6.188:27017'
dbname = 'highway'
tvswitch_tb = MongoClient(url)[dbname]['tvswitch']
tempimg_tb = MongoClient('localhost:27017')['highway']['temp_img']

class rtsp_reader:
    def __init__(self, cameraid):
        self.ipmap = {'TV%d'%i : 9+i*2 for i in range(1, 87)}
        self.cameraid = cameraid
        self.cap = self.get_cap()
        self.state = True
        if tempimg_tb.find({'cameraid':cameraid}).count() == 0:
            tempimg_tb.insert({'cameraid':cameraid, 'tempimg':'', 'time':time.time()})

    def get_cap(self):
        ip = self.ipmap[self.cameraid]
        url = 'rtsp://admin:a12345678@192.168.2.%d:554'%ip
        cap = cv2.VideoCapture(url)
   
        return cap

    def read(self):
        self.count = 0
        curt = time.time()
        
        state_last = True
        while True:
            self.count += 1
            ret, img = self.cap.read()
           
            #if not ret:
            #    print(self.cameraid,' no frame')
            #    break
            
            if time.time() - curt > 0.3:
                curt = time.time()
                if ret:
                    ret, buf = cv2.imencode('.jpg', img)
                    if ret:
                        jpg_as_text = base64.b64encode(buf)
                        tempimg_tb.update({'cameraid':self.cameraid}, {'$set':{'tempimg':jpg_as_text, 'time':curt}})
                        #cv2.imwrite(path, img)
                        print('update\t', str(curt))
                self.check_live()

    def check_live(self):
        while True:
            answer = list(tvswitch_tb.find({}))[0][self.cameraid]
            
            if answer == True :
                if self.state == False:
                    print('camera on')
                    self.cap = self.get_cap()
                self.state = True
                break
            else:
                if self.state == True:
                    print('camera die')
                
                self.cap = None
                self.state = False
                time.sleep(0.1)

if __name__ == '__main__':
    initcameraid = sys.argv[1]
    reader = rtsp_reader(initcameraid)
    reader.read()
