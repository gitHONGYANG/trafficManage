from urllib import request, parse

def upload(savepath, base64img):
    data = parse.urlencode(
        {'savepath':savepath, 'img64':base64img}
    ).encode('utf-8')
    req = request.Request('http://192.168.6.188:8082/saveimage', method='POST')
    with request.urlopen(req, data=data) as f:
        #print('Status:', f.status, f.reason)
        #print('Data:', f.read().decode('utf-8'))
        #print('upload ', savepath, 'status  %s\n' %f.status)
        pass
