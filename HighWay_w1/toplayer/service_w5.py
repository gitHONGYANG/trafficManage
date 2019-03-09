from flask import Flask, request
from utils.db_utils import find_visibility, find_car, find_object, search_rec_imgs
import json
import time
from datetime import datetime, date
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
from carmatch.car_match_full import MatchCar
import sys
from config import obj_tb, geo_tb, tvconfig_tb, tvconfig_history_tb
import os
import time

sys.path.append('carmatch/')
app = Flask(__name__, static_folder='/media/assests')


@app.route('/servertime', methods =['GET'])
def server_time():
    return str(time.time())


@app.route('/saveimage', methods=['POST'])
def save_images():
    #print('save_images')

    img64 = request.form.get('img64',None)
    savepath = request.form.get('savepath',None)
    img = Image.open(BytesIO(base64.b64decode(img64)))

    folder = savepath.split('/')[:-1]
    folder = '/'.join(folder)
    print('folder\t', folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print('makeflder\t', folder)
    
    img.save(savepath)

    return('Ok')

@app.route('/savevideo', methods=['POST'])
def save_videos():
    print('save_videos')
    video = request.files['video']    

    if video:
        filename = '/media/assests/Objvideos/' + video.filename
        video.save(filename)
    return ('ok')

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
        try:
            os.mkdir(_carmatchsavepath)
        except:
            pass
    #subpath in Recimgs, create folder every day
    _recimgsavepath = '/media/assests/Recimgs/' + datestr + '/'
    if not os.path.exists(_recimgsavepath):
        try:
            os.mkdir(_recimgsavepath)
        except:
            pass

    app.run(host='0.0.0.0', port=8085)
