from utils.change_tv import ChangeTV
import time
from config import tvswitch_tb


tvswitch_tb.remove({})

info = {'TV%d'%i: True if i%2==0 else False for i in range(3, 83)}
tvswitch_tb.insert_one(info)

tvs_alwayson = ['TV%d' %i for i in range(52, 72)]

c = ChangeTV()
while True:
    time.sleep(5)
    info_cur = list(tvswitch_tb.find({}))[0]
    info_new = {}
    for key in info_cur:
        if key in tvs_alwayson:
            info_new.update({key:True})

        elif info_cur[key] == True:
            info_new.update({key:False})
        elif info_cur[key] == False:
            info_new.update({key:True})
        
    print('\n----------', str(time.time()), '----------')
    #print(info_new)
    
    tvswitch_tb.update({}, {'$set':info_new})
