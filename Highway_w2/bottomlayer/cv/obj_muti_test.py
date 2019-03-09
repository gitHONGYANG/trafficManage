from core.service import Manager, LogReceiver
from provider.objvideoprovider import VideoProvider
from receiver.objreceiver_original import ObjDetecteReceiver
import time
import os
from core.config import obj_freq, obj_tb

#obj_tb.remove({})

manager = Manager()
path = '../../assests/ObjVideo/'
videos = os.listdir(path)
videos.sort()
videos = [path + video for video in videos]
videos = videos[:1]
print(videos)

providers = [VideoProvider(0, videos[i]) for i in range(len(videos))]

function2 = ObjDetecteReceiver(0, 'TV48')

function2.span(obj_freq)

for i in providers: manager.add_provider(i)

manager.add_receiver(function2)

manager.hold(True)
input()
