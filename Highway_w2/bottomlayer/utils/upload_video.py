import requests

def upload_video(videopath, filename):
    f = {'video':(filename, open(videopath, 'rb'))}
    req = requests.post('http://192.168.6.188:8082/savevideo', data = None, files=f)
    print('upload\t', filename)
if __name__ == '__main__':
    videopath = '../../../assests/Obj_videos/TV52_1531542011.0.avi' 
    upload_video(videopath, videopath.split('/')[-1])  
    print('done')
