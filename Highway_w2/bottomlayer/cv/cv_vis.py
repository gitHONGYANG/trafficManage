from core.service import Manager, LogReceiver
from core.config import carinfo_tb, _carmatchsavepath, obj_tb, vis_tb, rec_img_tb
from provider.videoprovider import VideoProvider
from receiver.carmatchreceiver import SaveCarInfo
from receiver.visreceiver import SaveVis
from receiver.objreceiver_original import ObjDetecteReceiver
from receiver.recimgreceiver import SaveRecentImg
import time
import os

vislist = [36, 37, 38, 40, 41, 42]

manager = Manager()
def build_provider():
    providers = [VideoProvider(i, 'TV%d' %(i)) for i in vislist]
    for i in providers:
        manager.add_provider(i)
def vis_receiver():
    for i in vislist:
        function = SaveVis(i, 'TV%d' %(i))
        function.span(60)
        manager.add_receiver(function)

def rec_receiver():
    for i in vislist:
        function = SaveRecentImg(i, 'TV%d' %(i))
        function.span(10)
        manager.add_receiver(function)

if __name__ == '__main__':
    #rec_img_tb.remove({})
    rec_receiver()
    
    #vis_tb.remove({})
    vis_receiver()
    #obj_receiver()

    build_provider()
    manager.hold(True)
    input()
