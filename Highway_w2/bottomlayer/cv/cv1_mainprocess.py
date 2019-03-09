from core.service import Manager, LogReceiver
from core.config import carinfo_tb, _carmatchsavepath, obj_tb, vis_tb, rec_img_tb, car_tb, tvconfig_tb
from provider.videoprovider import VideoProvider
from receiver.carmatchreceiver import SaveCarInfo
from receiver.visreceiver import SaveVis
from receiver.objreceiver_original import ObjDetecteReceiver
from receiver.recimgreceiver import SaveRecentImg
import time
import os

info = list(tvconfig_tb.find({'name':'w5'}))[0]
tvs = info['tvs']
tvs = tvs[::5]

print('tvlist\t', str(tvs))

def clean_tbs():
    carinfo_tb.remove({})
    obj_tb.remove({})
    vis_tb.remove({})
    rec_img_tb.remove({})

manager = Manager()
def build_provider():
    providers = [VideoProvider(i, tv) for i, tv in enumerate(tvs)]
    #providers = [VideoProvider(i, 'TV%d' %(57+i)) for i in range(7)]
    for i in providers:
        manager.add_provider(i)

def carmatch_receiver():
    for i, tv in enumerate(tvs):
        picpath = _carmatchsavepath + tv
        TVID = tv
        function = SaveCarInfo(i, picpath, TVID)
        function.span(0.3)
        manager.add_receiver(function)

def obj_receiver():
    for i, tv in enumerate(tvs):
        function = ObjDetecteReceiver(i, tv)
        function.span(0.8)
        manager.add_receiver(function)

def vis_receiver():
    for i, tv in enumerate(tvs):
        function = SaveVis(i, tv)
        function.span(60)
        manager.add_receiver(function)

def rec_receiver():
    for i, tv in enumerate(tvs):
        function = SaveRecentImg(i, tv)
        function.span(10)
        manager.add_receiver(function)

def car_test_receiver():
    lis = [46]
    for i,l in enumerate(lis):
        picpath = _carmatchsavepath +'TV%d/' %(l)
        TVID = 'TV%d' %(l)
        function = SaveCarInfo(i, picpath, TVID)
        function.span(0.5)
        manager.add_receiver(function)

def obj_test_receiver():
    lis = [46]
    for i,l in enumerate(lis):
        function = ObjDetecteReceiver(i, 'TV%d' %(l))
        function.span(0.8)
        manager.add_receiver(function)

def car_test_provider():
    lis = [46]
    #providers = [VideoProvider(i, 'TV%d' %(l)) for i,l in enumerate(lis)]
    providers = [VideoProvider(i, '/media/assests/CarVideo/TV46_output.mp4') for i,l in enumerate(lis)]
    for i in providers:
        manager.add_provider(i)


if __name__ == '__main__':
    #rec_img_tb.remove({})
    #rec_receiver()
    
    vis_tb.remove({})
    vis_receiver()
    #obj_receiver()
    
    #carinfo_tb.remove({})
    #carinfo_tb.remove({})
    #carmatch_receiver()

    #car_tb.remove({})
    #car_test_receiver()
    #obj_test_receiver()
    #car_test_provider()
    
    build_provider()
    manager.hold(True)
    time.sleep(24*60*3600)
