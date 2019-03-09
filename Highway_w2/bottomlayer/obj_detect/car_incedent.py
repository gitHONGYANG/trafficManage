import os
import datetime
from pymongo import MongoClient

url = '192.168.6.188'
dbname = 'highway'

date = datetime.datetime.now().date()
datestr = str(date.year)+'-'+str(date.month)+'-'+str(date.day)
car_name = 'car_' + datestr
print(car_name)
car_tb = MongoClient(url)[dbname][car_name]


infos = list(car_tb.find({}))
infos = infos[::-1]

warnningt = 0
for info in infos:
    t = info['time']
    if t < warnningt + 30:
        continue
    cameraid = info['cameraid']
    anchor_location = info['location']
    where = {'time':{'$gt':t-30,'$lt':t}, 'cameraid':cameraid}
    cars = list(car_tb.find(where))
    
    anchor_centery = int((anchor_location[3]-anchor_location[1])/2 + anchor_location[1])
    anchor_centerx = int((anchor_location[2]-anchor_location[0])/2 + anchor_location[0])
    anchor_w = int(anchor_location[2] - anchor_location[0])
    anchor_h = int(anchor_location[3] - anchor_location[1])

    shift_w = int(anchor_w/10)
    shift_h = int(anchor_h/10)

    count = 0
    for car in cars:
        location = car['location']
        
        centery = int((location[3] - location[1])/2 + location[1])
        centerx = int((location[2] - location[0])/2 + location[0])
        
        if abs(centerx - anchor_centerx)<shift_w and abs(centery - anchor_centery)<shift_h:
            count += 1
    if count > 10:
        print(info)
        warnningt = t
