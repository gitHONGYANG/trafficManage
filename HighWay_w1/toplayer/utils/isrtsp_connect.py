import time
import cv2
import sys
from pymongo import MongoClient

url = 'mongodb://localhost:27017'
dbname = 'highway'
iswork_tb = MongoClient(url)[dbname]['iswork']


#tempimg_tb.update({'cameraid':cameraid}, {'$set':{'tempimg':jpg_as_text, 'count':saveimg_count, 'time': curt, 'frame_process':frame_process}})
def rtsp_basetest(camid):
    ip = int(camid.split('V')[-1]) * 2 +9
    initt = time.time()

    info = {'cameraid':camid, 'time':time.time(), 'state':'check'}
    if len(list(iswork_tb.find({'cameraid':camid}))) == 0:
        iswork_tb.insert(info)
    else:
        iswork_tb.update({'cameraid':camid}, {'$set':info})

    try:
        cap = cv2.VideoCapture('rtsp://admin:a12345678@192.168.2.%d:554' %ip)
    except:
        info = {'cameraid':camid, 'time':time.time()+5, 'state':'False'}
        if len(list(iswork_tb.find({'cameraid':camid}))) == 0:
            iswork_tb.insert(info)
        else:
            iswork_tb.update({'cameraid':camid}, {'$set':info})
        return False
    counter = 0
    while True:
        ret, frame = cap.read()
        counter += 1
        if ret:
            if counter > 5:
                info = {'cameraid':camid, 'time':time.time(), 'state':'True'}
                if len(list(iswork_tb.find({'cameraid':camid}))) == 0:
                    iswork_tb.insert(info)
                else:
                    iswork_tb.update({'cameraid':camid}, {'$set':info})
                return True
            if time.time() - initt >5:
                info = {'cameraid':camid, 'time':time.time(), 'state':'False'}
                if len(list(iswork_tb.find({'cameraid':camid}))) == 0:
                    iswork_tb.insert(info)
                else:
                    iswork_tb.update({'cameraid':camid}, {'$set':info})
                return False
        else:
            info = {'cameraid':camid, 'time':time.time(), 'state':'False'}
            if len(list(iswork_tb.find({'cameraid':camid}))) == 0:
                iswork_tb.insert(info)
            else:
                iswork_tb.update({'cameraid':camid}, {'$set':info})
            return False

if __name__ == '__main__':
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    camids = ['TV%d' %i for i in range(start, end + 1)] 
    for camid in camids:
       print(rtsp_basetest(camid))
