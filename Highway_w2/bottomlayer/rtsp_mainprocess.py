from core.service import Manager, LogReceiver   #登录？？
from core.config import carinfo_tb, _carmatchsavepath, obj_tb, vis_tb, rec_img_tb, car_tb, tvconfig_tb  #配置文件
from provider.rtspprovider import RtspProvider   #获得数据流
from receiver.carmatchreceiver import SaveCarInfo    #车辆匹配
from receiver.visreceiver import SaveVis
from receiver.objreceiver_original import ObjDetecteReceiver
from receiver.recimgreceiver import SaveRecentImg
import time
import os
import sys

tvs = ['TV4', 'TV5', 'TV6', 'TV7', 'TV8', 'TV9', 'TV10', 'TV11', 'TV12', 'TV13']

i = int(sys.argv[1])   #得到外部给定的第一个参数，选择摄像头
if i == 0:
    tvs = tvs[0:3] + ['TV52']
elif i == 1:
    tvs = tvs[3:6] + ['TV53']
elif i == 2:
    tvs = tvs[6:9] + ['TV54'] 
elif i == 3:
    tvs = tvs[9:10] + ['TV55', 'TV56']

print('tvlist\t', str(tvs)) #结果输出

def clean_tbs():
    #清空什么？？？
    carinfo_tb.remove({})
    obj_tb.remove({})
    vis_tb.remove({})
    rec_img_tb.remove({})

manager = Manager()
def build_provider():
    providers = [RtspProvider(i, tv) for i, tv in enumerate(tvs)] #from provider.rtspprovider import RtspProvider 得到服务器
    for i in providers:
        manager.add_provider(i)

def carmatch_receiver():
    for i, tv in enumerate(tvs):
        picpath = _carmatchsavepath + tv         #图片路径
        TVID = tv
        function = SaveCarInfo(i, picpath, TVID)  #from receiver.carmatchreceiver import SaveCarInfo     保存图片的相关信息
        function.span(0.6)
        manager.add_receiver(function)

def obj_receiver():
    for i, tv in enumerate(tvs):
        function = ObjDetecteReceiver(i, tv)   #from receiver.objreceiver_original import ObjDetecteReceiver    物体探测
        function.span(1)
        manager.add_receiver(function)

def vis_receiver():
    for i, tv in enumerate(tvs):
        function = SaveVis(i, tv)          #from receiver.visreceiver import SaveVis   作用应该是保存录像
        function.span(30)
        manager.add_receiver(function)

def rec_receiver():
    for i, tv in enumerate(tvs):
        function = SaveRecentImg(i, tv)   #from receiver.recimgreceiver import SaveRecentImg  作用应该是保存刚刚获得的图片
        function.span(10)
        manager.add_receiver(function)   #将摄像头对应的接收器放到manager中进行管理

if __name__ == '__main__':
    rec_receiver()
    vis_receiver()
    carmatch_receiver()
    obj_receiver()
    build_provider()
    manager.hold(True)
    time.sleep(24*60*3600)   #休眠一天的作用是？？？
