from core.service import ImageReceiver
import time
import datetime
import uuid
from core.config import rec_img_tb
import cv2

class SaveRecentImg(ImageReceiver):
    def __init__(self, ID, savepicpath, cameraid):
        self.id = ID
        self.info = None

        self.savepicpath = savepicpath
        if not os.path.exists(self.savepicpath)
            os.mkdir(self.savepicpath)
            print('create folder ', self.savepicpath)

        self.cameraid = cameraid

    def feed(self, info):
        frame = info['img']

        imageid = str(uuid.uuid1())
        t = info['time']
        t = float(time.mktime(datetime.datetime(2018, 7, 14, 12, 0, 0).timetuple()) + t)
        cameraid = self.cameraid
        imagepath = self.savepicpath + cameraid + '_' + str(t) + '.jpg'

        frame = cv2.resize(frame, (1024,768))
        cv2.imwrite(imagepath, frame, [int(cv2.IMWRITE_JPEG_QULITY), 50])
        recent_img_info = {
                'imageid':imageid,
                'time':t,
                'cameraid':cameraid,
                'imagepath':imagepath,
                'info':''
                }

        rec_img_tb.insert_one(recent_img_info)
