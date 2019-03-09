import pymongo
from pymongo import MongoClient
import pandas as pd
from datetime import datetime,date
import json
from time import mktime
from config import vis_tb, obj_tb, rec_img_tb, car_match_tb

client = MongoClient('localhost', 27017)
#obj_tb.remove({})
#vis_tb.remove({})
#car_match_tb.remove({})

def search_rec_imgs(start=None, end=None, camid=None, limit = None, unique = None):
    today = datetime.now()
    today_start = float(mktime(datetime(today.year, today.month, today.day, 0).timetuple()))
    today_end = float(mktime(datetime(today.year, today.month, today.day, 23, 59).timetuple()))

    start = start or today_start
    start = float(start)
    end = end or today_end
    end = float(end)
    limit = limit or 10000
    unique = unique or False

    t = date.fromtimestamp(start)
    tbname = 'recent_' + str(t.year) + '-' + str(t.month) + '-' + str(t.day)
    rec_tb = client['highway'][tbname]

    where = {'time':{'$gt':start, '$lt':end}}

    if camid: where['cameraid'] = camid
    rst = list(rec_tb.find(where).limit(limit))
    
    if not unique: return rst
    uniques = {}
    for img in rst:
        if not img['cameraid'] in uniques:
            uniques[img['cameraid']] = img
    return list(uniques.values())

def find_visibility(start=None, end=None, saferank=None, camid=None, limit = None, unique = None):
    today = datetime.now()
    today_start = float(mktime(datetime(today.year, today.month, today.day, 0).timetuple()))
    today_end = float(mktime(datetime(today.year, today.month, today.day, 23, 59).timetuple()))

    start = start or today_start
    start = float(start)
    end = end or today_end
    end = float(end)
    limit = limit or 10000
    unique = unique or False

    where = {'time':{'$gt':start, '$lt':end}}
    if not saferank is None: where['saferank'] = saferank
    if not camid is None: where['cameraid'] = camid
    
    print('-----------')
    print(where)
    
    rst = list(vis_tb.find(where).limit(limit))
    
    if not unique: return rst
    uniques = {}
    for vis in rst:
        if not vis['cameraid'] in uniques:
            uniques[vis['cameraid']] = vis
    return list(uniques.values())

#need fix about mongodb
def find_car(start=None, end=None, camid=None):
    today = datetime.now()

    today_start = float(mktime(datetime(today.year, today.month, today.day, 0).timetuple()))
    today_end = float(mktime(datetime(today.year, today.month, today.day, 23, 59).timetuple()))

    start = start or today_start
    start = float(start)
    end = end or today_end
    end = float(end)
    
    t = date.fromtimestamp(start)
    tbname = 'carmatch_' + str(t.year) + '-' + str(t.month) + '-' + str(t.day)
    print('tbname\t', str(tbname))

    carmatch_tb = client['highway'][tbname]

    where = {'time':{'$gt':start, '$lt':end}}
    if start == mktime(datetime(t.year, t.month, t.day, 0,0,0).timetuple()) and end == mktime(datetime(t.year, t.month, t.day, 23,59,59).timetuple()):
        where = {}
        print('where is none')
    if not camid is None: where['cameraid'] = camid

    return list(carmatch_tb.find(where))


def find_object(start=None, end=None, saferank=None, state=None , camid=None):
    today = datetime.now()
    today_start = float(mktime(datetime(today.year, today.month, today.day, 0).timetuple()))
    today_end = float(mktime(datetime(today.year, today.month, today.day, 23, 59).timetuple()))

    start = start or today_start
    start = float(start)
    end = end or today_end
    end = float(end)

    where = {'time':{'$gt':start, '$lt':end}}
    if not saferank is None: where['saferank'] = saferank
    if not camid is None: where['cameraid'] = camid
    if not state is None: where['status'] = state

    return list(obj_tb.find(where))

if __name__ == '__main__':

    start_time = datetime(2018, 7, 14, 12, 0, 0)
    start_time = time.mktime(start_time.timetuple())
    start_time = start_time

    end_time = datetime(2018, 7, 14, 13, 0, 0)
    end_time = time.mktime(end_time.timetuple())
    end_time = end_time

    visibility_infos_json = find_visibility(start_time, end_time, None, None)
