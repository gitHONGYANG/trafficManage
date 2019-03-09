from core.service import ImageReceiver
from obj_detect.objdetect import HighWayObjDetect
import time
import datetime
from core.config import obj_tb


class ObjDetecteReceiver(ImageReceiver):
    #caneraud : 'TV36' e.g.
    def __init__(self, id, cameraid):
        self.info = None
        self.id = id
        self.cameraid = cameraid
        #self.objdetect = HighWayObjDetect(cameraid, modelpath)
        self.objdetect = HighWayObjDetect(cameraid, '../../checkpoints/alex_epoch9.pt')

    def feed(self, info):
        img = info['img']
        t = float(time.mktime(datetime.datetime(2018,7,14,12,0,0).timetuple()) + info['time'])
        self.objdetect.step(img, t)
