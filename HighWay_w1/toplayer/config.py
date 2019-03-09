#local ip 192.168.6.188
import datetime
import pymongo
from pymongo import MongoClient

url = 'mongodb://localhost:27017'
dbname = 'highway'

vis_tb = MongoClient(url)[dbname]['visibility']

obj_tb = MongoClient(url)[dbname]['object']

date = datetime.datetime.now().date()

geo_tb = MongoClient(url)[dbname]['geoconfig']

tvconfig_tb = MongoClient(url)[dbname]['tvconfig']

tvconfig_history_tb = MongoClient(url)[dbname]['tvconfig_history']

tvswitch_tb = MongoClient(url)[dbname]['tvswitch']

'''
recnetimages table, collection name is like recent_2018-7-14
'''
rectb_name = 'recent_' + str(date.year) + '-' + str(date.month) + '-' + str(date.day)
#rectb_name = 'recent_' + str(2018) + '-' + str(10) + '-' + str(2)
rec_img_tb = MongoClient(url)[dbname][rectb_name]

#print(rectb_name)

rec_img_tb.create_index([('time', pymongo.DESCENDING)], unique = False)

'''
carmatchtb , collection name is like carmach_2018-7-14
'''
carmatchtb_name = 'carmatch_' + str(date.year) + '-' + str(date.month) + '-' + str(date.day)
#carmatchtb_name = 'carmatch_' + str(2018) + '-' + str(7) + '-' + str(14)
car_match_tb = MongoClient(url)[dbname][carmatchtb_name]
car_match_tb.create_index([('time', pymongo.DESCENDING)], unique = False)


cartb_name = 'car_' + str(date.year) + '-' + str(date.month) + '-' + str(date.day)
car_tb = MongoClient(url)[dbname][cartb_name]
car_tb.create_index([('time', pymongo.DESCENDING)], unique = False)

visi_original = 'visibility'
visi_original_tb = MongoClient(url)[dbname][visi_original]
visi_original_tb.create_index([('time', pymongo.DESCENDING)], unique = False)



visi_out = 'visibility_out'
visi_out_tb = MongoClient(url)[dbname][visi_out]
visi_out_tb.create_index([('time', pymongo.DESCENDING)], unique = False)

#不同段能见带
zmd = ['TV%d' %i for i in range(1, 12)]
qs = ['TV%d' %i for i in range(12, 25)]
mg = ['TV%d' %i for i in range(25, 40)]
hd = ['TV%d' %i for i in range(40, 56)]
xy = ['TV%d' %i for i in range(56, 69)]
ls = ['TV%d' %i for i in range(69, 76)]
sj = ['TV%d' %i for i in range(76, 86)]

if __name__ == '__main__':
    #vis_tb.remove({'time':{'$gt':1541001600}})
    tvconfig_tb.insert({'name':'w2', 'tvlist':['TV%d'%i for i in range(36, 43)]})
    tvconfig_tb.insert({'name':'w3', 'tvlist':['TV%d'%i for i in range(43, 50)]})
    tvconfig_tb.insert({'name':'w4', 'tvlist':['TV%d'%i for i in range(50, 57)]})
    tvconfig_tb.insert({'name':'w5', 'tvlist':['TV%d'%i for i in range(57, 64)]})
