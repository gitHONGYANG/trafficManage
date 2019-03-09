#service脚本定义了诸多基准类
import numpy as np
from threading import Thread
from multiprocessing import Process, Pool
from time import time, sleep
import pdb

class ImageProvider:
    '''
    作为诸多provider的父类
    '''
    def __init__(self, id):
        self.id = id
    def impulse(self): return None
    def ok(self): return True

    def hold(self, handle=lambda x:x):
        #等待
        while True:
            sleep(0.01)
            if not self.ok():
                print('privider', self.id, 'not work')
                sleep(0.99)
                continue
            handle(self.impulse())

class ImageReceiver:
    def __init__(self, id):
        self.id = id
        self.info = None

    def span(self, span):
        self.sep = span    #保存需要间隔的时间
        self.now = 0

    def timeon(self, now):
        if not hasattr(self, 'sep'):     #判断self 是否有 sep 属性
            return True
        if now-self.now>self.sep:      #当间隔时间达到一定值
            self.now = now
            return True
        #这个函数确定没有问题吗？？？？？？？

    def feed(self, info): pass    #这个函数一般由子类重写
    def feed_asyn(self, info):    #将参数信息存储
        self.info = info
    def ok(self): return True    #这个函数意思？？？

    def listen(self):           #监听机制
        print('listening')
        while True:
            sleep(0.01)
            if not self.info is None:   #info值非空
                # print(self.timeon(self.info['time']))
                if self.timeon(self.info['time']):
                    #print('Receiver %s feeded at time %.1f'%(self.id, self.info['time']))
                    self.feed(self.info)
                self.info = None

class RandomProvider(ImageProvider):
    def impulse(self):
        img = np.random.rand(300*300).reshape((300,300))
        return {'id':self.id, 'img': img, 'time':time()}

class LogReceiver(ImageReceiver):
    def feed(self, info):
        print('ID:', info['id'], info['img'].mean())

class Manager:
    def __init__(self):
        self.provider = []   #服务器群
        self.receiver = []   #客户端群

    def add_provider(self, pro):
        self.provider.append(pro)

    def add_receiver(self, rec):
        self.receiver.append(rec)

    def step(self, asyn=False):
        for i in self.provider:
            if not i.ok():
                print('privider', i.id, 'not work')
                continue
            info = i.impulse()
            for j in self.receiver:
                if j.id != info['id']:continue
                if not asyn and j.timeon(info['time']):
                    print('Receiver %s feeded at time %.1f'%(j.id, info['time']))
                    j.feed(info)
                else: j.feed_asyn(info)

    def wait(self):
        while sum([not i.info is None for i in self.receiver])>0:continue

    def hold(self, autoread=True):
        '''
        pool = Pool(len(self.receiver))
        for i in self.receiver:
                pool.apply_async(i.listen)
        '''
        ths = [Thread(target=i.listen) for i in self.receiver]  #为每个receiver创造线程
        for i in ths: i.daemon = True    #如果某个子线程的daemon属性为True，主线程运行结束时不对这个子线程进行检查而
        # 直接退出，同时所有daemon值为True的子线程将随主线程一起结束，而不论是否运行完成。
        for i in ths: i.start()
        if not autoread: return
        def feed(info):
            for i in self.receiver:
                if i.id==info['id']: i.feed_asyn(info)
        ths = [Thread(target=i.hold, args=(feed,)) for i in self.provider]  #为每个服务器创作线程，将service的动作存贮
        for i in ths: i.daemon = True
        for i in ths: i.start()

if __name__ == '__main__':
    manager = Manager()
    provider1 = RandomProvider(1)
    provider2 = RandomProvider(2)
    receiver1 = LogReceiver(1)
    receiver2 = LogReceiver(2)
    receiver1.span(1)
    receiver2.span(1)
    manager.add_provider(provider1)
    manager.add_receiver(receiver1)
    manager.add_provider(provider2)
    manager.add_receiver(receiver2)

    manager.hold(True)

