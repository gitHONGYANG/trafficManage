from flask import Flask, request    #flask  web框架的一种
from utils.db_utils import find_visibility, find_car, find_object, search_rec_imgs
import json   #json文件相关库
import time
from datetime import datetime, date  #日期
import base64     #编码
from io import BytesIO
from PIL import Image
import pandas as pd
from carmatch.car_match_full import MatchCar
import sys
from config import obj_tb, geo_tb, tvconfig_tb, tvconfig_history_tb
import os
import time

sys.path.append('carmatch/')  #加入系统目录
app = Flask(__name__, static_folder='/media /assests')
matchcarprocess = MatchCar()    #匹配车的进程

#default search for 30 seconds
#save one pic every 10s
#return recent images of every camera, so maxnum is 28, if not , show XX camera not work in client
@app.route('/recentsearch', methods =['GET'])
def search_rec_img():
    paras = ['start', 'end', 'cameraid', 'limit', 'unique']
    args = [request.args.get(i,None) for i in paras]
    if not args[3] is None: args[3] = int(args[3])
    if not args[4] is None: args[4] = eval(args[4])
    
    rec_imgs = search_rec_imgs(*args)
    rst = []
    for info in rec_imgs:
        info.pop('_id')
        rst.append(info)
        
    return json.dumps(rst)

@app.route('/servertime', methods =['GET'])
def server_time():
    return str(time.time())

@app.route('/iswork', methods = ['GET'])
def iswork():
    print('rtsp check')
    os.system('/home/highway/HighWay/toplayer/utils/isrtsp_connect.sh')
    return 'ok'

@app.route('/visibility', methods= ['GET'])
def get_visibility(lst=[]):
    paras = ['start', 'end', 'saferank', 'cameraid', 'limit', 'unique']
    args = [request.args.get(i,None) for i in paras]
    if not args[2] is None: args[2] = int(args[2])
    if not args[4] is None: args[4] = int(args[4])
    if not args[5] is None: args[5] = eval(args[5])
    vis_infos = find_visibility(*args)

    rst = []
    for info in vis_infos:
        info.pop('_id')
        rst.append(info)

    return json.dumps(rst)

#start end none need fix
@app.route('/object', methods= ['GET'])
def get_object(lst=[]):
    paras = ['start', 'end', 'saferank', 'status', 'cameraid']
    args = [request.args.get(i,None) for i in paras]

    start = args[0]
    end = args[1]

    startdate = datetime.fromtimestamp(float(start))
    enddate = datetime.fromtimestamp(float(end))
    print('start date', str(startdate))
    print('end date', str(enddate))

    if not args[2] is None: args[2] = int(args[2])
    if not args[3] is None: args[3] = int(args[3])
    vis_infos = find_object(*args)

    rst = []
    for info in vis_infos:
        info.pop('_id')
        rst.append(info)

    return json.dumps(rst)

@app.route('/update_objstatus', methods = ['GET'])
def update_objstate():
    objid = request.args.get('obj_id', None)
    newstate = request.args.get('status', None)
    if objid and newstate:
        obj_tb.update_one({'objid':objid}, {"$set":{'state':int(newstate)}})
        return 'success'
    else:
        return 'fail'

@app.route('/getcars', methods = ['GET'])
def get_cars():
    start = request.args.get('start',None)
    end = request.args.get('end', None)
    cameraid = request.args.get('camid', None)

    cars = list(find_car(start=start, end=end, camid=cameraid))
    rst = []
    for info in cars:
        rst.append({'imagepath':info['imagepath'], 'time':info['time'], 'cameraid':info['cameraid']})

    return json.dumps(rst)


@app.route('/matchcar', methods=['POST'])
def get_samecar():
    initt = time.time()

    t = float(request.form.get('time',None))
    camid = request.form.get('camid',None)
    img64 = request.form.get('img64',None)

    img = Image.open(BytesIO(base64.b64decode(img64)))

    print('\t\t\topenimg\t', str(time.time()-initt))
    initt = time.time()

    da = date.fromtimestamp(t)
    #start_t = time.mktime(datetime(da.year, da.month, da.day).timetuple())
    start_t = t - 3600*2
    end_t = time.mktime(datetime(da.year, da.month, da.day, 23, 59, 59).timetuple())
    df = pd.DataFrame(find_car(start_t, end_t))

    print('builddf\t', str(time.time()-initt))


    initt = time.time()
    car_infos = matchcarprocess.searchcar(img, t, camid, df)

    print('searchcar\t', str(time.time()-initt))

    initt = time.time()
    rst = []
    for carinfo in car_infos:
        #geoinfo = geo_tb.find_one({'name':carinfo[4]})
        #dk = geoinfo['dk']
        #position = geoinfo['position']
        #print(carinfo)

        returncar = {'time':carinfo[3], 'imagepath':carinfo[2], 'cameraid':carinfo[4], 'position':carinfo[6]}
        rst.append(returncar)

    print('returnjson\t', str(time.time()-initt))

    return json.dumps(rst)

if __name__ == '__main__':
    date = datetime.now().date()
    datestr = str(date.year)+'-'+str(date.month)+'-'+str(date.day)

    #assests rootdir,create them if not exists
    rootdirs = ['/media/assests/Objimgs_rawimgs', '/media/assests/Recimgs', '/media/assests/Cars']
    for rtdir in rootdirs:
        if not os.path.exists(rtdir):
            os.mkdir(rtdir)

    #subpath in Cars, create folder every day
    _carmatchsavepath = '/media/assests/Cars/' + datestr + '/'
    if not os.path.exists(_carmatchsavepath):
        os.mkdir(_carmatchsavepath)

    #subpath in Recimgs, create folder every day
    _recimgsavepath = '/media/assests/Recimgs/' + datestr + '/'
    if not os.path.exists(_recimgsavepath):
        os.mkdir(_recimgsavepath)
    
    app.run(host='0.0.0.0', port=8080)
