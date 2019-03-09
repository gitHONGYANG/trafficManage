from core.service import ImageReceiver
from obj_detect.objdetect_original import HighWayObjDetect
import time
import datetime
from core.config import obj_tb


class ObjDetecteReceiver(ImageReceiver):
    #caneraud : 'TV36' e.g.
    def __init__(self, id, cameraid):
        self.info = None
        self.id = id
        self.cameraid = cameraid
        self.objdetect = HighWayObjDetect(cameraid, '/media/assests/checkpoints/obj_resnet_29.pt')
        
        #self.time = 0 


    def feed(self, info):
        img = info['img']
        t = info['time']
        '''
        self.time = info['time']
        tempt = datetime.datetime(2018,7,30,10,32, 58) + datetime.timedelta(seconds=self.time)
        t = time.mktime(tempt.timetuple())
        print('receive %f' %t)
        '''
        self.objdetect.step(img, t)
