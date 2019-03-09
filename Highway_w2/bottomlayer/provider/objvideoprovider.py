from core.service import ImageProvider
import cv2
import datetime
import time


class VideoProvider(ImageProvider):
    def __init__(self, id, path):
        self.id = id
        self.cap = cv2.VideoCapture(path)
        self.datetime = datetime.datetime(2018, 10, 14, 12, 0, 0)
        self.time = time.mktime(self.datetime.timetuple())

    def impulse(self):
        #self.time += 1 / 25
        return {'id': self.id, 'img': self.frame, 'time': self.time}

    def ok(self):
        self.status, self.frame = self.cap.read()
        #self.datetime = self.datetime + datetime.timedelta(seconds = 1/25)
        #self.time = time.mktime(self.datetime.timetuple())
        self.time += 1/25
        if not self.status: print('video end')
        return self.status
