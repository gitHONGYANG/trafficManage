import datetime
import pymongo
from pymongo import MongoClient
import os

date = datetime.datetime.now().date()

#datestr = str(2018)+'-'+str(7)+'-'+str(11)
datestr = str(date.year)+'-'+str(date.month)+'-'+str(date.day)

#assests rootdir,create them if not exists
rootdirs = ['/meida/assests/Objimgs', '/media/assests/Objimgs_rawimgs', '/media/assests/Recimgs', '/media/assests/Cars']

#subpath in Cars, create folder every day
_carmatchsavepath = '/media/assests/Cars/' + datestr + '/'

#subpath in Recimgs, create folder every day
_recimgsavepath = '/media/assests/Recimgs/' + datestr + '/'

#import them when in receiver
_objimgsavepath = '/media/assests/Objimgs/'
_objrawimgsavepath = '/media/assests/Objimgs_rawimgs/'

#carmatch tvs
tvs_alwayson = ['TV%d' %i for i in range(52, 72)]

#model path
vismodelpath = '/media/assests/checkpoints/vis_regression_resnet18_9_.pt'
yolomodelpath = '/media/assests/checkpoints/yolo_exp3_66.pkl'
similaritymodelpath = '/media/assests/checkpoints/carMatch_CCL_epoch10_305000.pt'
objmodelpath = '/media/assests/checkpoints/obj_resnet_29.pt'

#database
url = 'mongodb://192.168.6.188:27017'
dbname = 'highway'
obj_tb = MongoClient(url)[dbname]['object']
accident_tb = MongoClient(url)[dbname]['accident']

vis_tb = MongoClient(url)[dbname]['visibility']

rectb_name = 'recent_' + datestr
rec_img_tb = MongoClient(url)[dbname][rectb_name]

carinfo_name = 'carmatch_' + datestr
carinfo_tb = MongoClient(url)[dbname][carinfo_name]

car_name = 'car_' + datestr
car_tb = MongoClient(url)[dbname][car_name]

tvconfig_tb = MongoClient(url)[dbname]['tvconfig']
#log
log_db = MongoClient('localhost:27017')['log']

#config
vis_freq = 30
rec_freq = 10
obj_freq = 0.8

#
camera_rtsp = {}
for i in range(1, 87):
    camera_id = 'TV%d' %(i)
    rtsp_url = 'rtsp://admin:a12345678@192.168.2.%d:554' %(i*2+9)
    camera_rtsp[camera_id] = rtsp_url

geo_loc = {
    'TV1' : [874, 948],
    'TV2' : [876, 425],
    'TV3' : [878, 606],
    'TV4' : [879, 450],
    'TV5' : [880, 400],
    'TV6' : [882, 276],
    'TV7' : [884, 425],
    'TV8' : [886, 425],
    'TV9' : [888, 430],
    'TV10': [890, 430],
    'TV11': [891, 830],
    'TV12': [893, 230],
    'TV13': [895, 225],
    'TV14': [897, 225],
    'TV15': [898, 230],
    'TV16': [900, 25],
    'TV17': [900, 730],
    'TV18': [902, 615],
    'TV19': [904, 635],
    'TV20': [906, 659],
    'TV21': [908, 630],
    'TV22': [910, 630],
    'TV23': [912, 630],
    'TV24': [914, 625],
    'TV25': [915, 625],
    'TV26': [916, 630],
    'TV27': [917, 715],
    'TV28': [918, 625],
    'TV29': [920, 630],
    'TV30': [922, 30],
    'TV31': [924, 775],
    'TV32': [926, 850],
    'TV33': [928, 950],
    'TV34': [930, 940],
    'TV35': [932, 530],
    'TV36': [934, 530],
    'TV37': [936, 500],
    'TV38': [937, 530],
    'TV39': [938, 530],
    'TV40': [939, 550],
    'TV41': [940, 530],
    'TV42': [942, 540],
    'TV43': [943, 550],
    'TV44': [944, 550],
    'TV45': [945, 520],
    'TV46': [946, 530],
    'TV47': [947, 530],
    'TV48': [948, 530],
    'TV49': [950, 540],
    'TV50': [952, 530],
    'TV51': [954, 0],
    'TV52': [954, 950],
    'TV53': [956, 950],
    'TV54': [959, 950],
    'TV55': [960, 930],
    'TV56': [961, 930],
    'TV57': [962, 950],
    'TV58': [964, 880],
    'TV59': [966, 860],
    'TV60': [968, 300],
    'TV61': [969, 730],
    'TV62': [970, 620],
    'TV63': [972, 0],
    'TV64': [973, 30],
    'TV65': [975, 0],
    'TV66': [976, 820],
    'TV67': [978, 230],
    'TV68': [980, 0],
    'TV69': [980, 900],
    'TV70': [982, 800],
    'TV71': [984, 200],
    'TV72': [985, 300],
    'TV73': [986, 380],
    'TV74': [987, 500],
    'TV75': [988, 450],
    'TV76': [990, 140],
    'TV77': [992, 220],
    'TV78': [993, 690],
    'TV79': [994, 690],
    'TV80': [996, 370],
    'TV81': [1000, 200],
    'TV82': [1002, 80],
    'TV83': [1003, 80],
    'TV84': [1004, 80],
    'TV85': [1005, 80],
    'TV86': [1006, 80],
}

if __name__ == '__main__':
    print(camera_rtsp)

