import cv2
import base64
import pymongo
from pymongo import MongoClient
import datetime
import time
import sys

tempimg_tb = MongoClient('localhost:27017')['highway']['temp_img']
#tempimg_tb.remove({})
def cv_reader(path, cameraid):
    if tempimg_tb.find({'cameraid':cameraid}).count() == 0:
        tempimg_tb.insert({'cameraid':cameraid, 'tempimg':'', 'count':0})
        print('%s db init' %cameraid)

    cap = cv2.VideoCapture(path)
    frame_count = 0
    saveimg_count = 0
    initt = datetime.datetime(2018, 7, 14, 12, 0, 0)
    initt = time.mktime(initt.timetuple())

    initt_process = time.time()

    t_save = time.time()
    while True:
        #time.sleep(0.025)
        ret, img = cap.read()
        if not ret:
            print(cameraid,' no frame')
            break
        frame_count += 1
        
        t_cur = time.time()
        if t_cur - t_save > 0.2:
            t_save = t_cur
            ret, buffer = cv2.imencode('.jpg', img)
            jpg_as_text = base64.b64encode(buffer)
            saveimg_count += 1
            t = initt + frame_count/25
            
            frame_process = frame_count/(time.time() - initt_process)

            tempimg_tb.update({'cameraid':cameraid}, {'$set':{'tempimg':jpg_as_text, 'count':saveimg_count, 'time': t, 'frame_process': frame_process}})
        if frame_count%(10*60*25) == 0 and saveimg_count!=0:
            t = time.localtime(time.time())
            t = time.strftime("%Y-%m-%d %H:%M:%S", t)
            print(cameraid+'\t'+ t + '\tvideotime  %ds' %int(frame_count/25) +'\tframecount  %d' %frame_count)

if __name__ == '__main__':
    cameraid = sys.argv[1]
    path = '/media/assests/CarVideo/%s_12_14.mp4' %cameraid
    #path = '/media/assests/CarVideo/TV46_output.mp4'
    cv_reader(path, cameraid)
