from config import visi_original_tb, visi_out_tb
from config import zmd, qs, mg, hd, xy, ls, sj
import time

while True:
    start = time.time() - 60
    end = time.time()
    where = {'time':{'$gt' : start, '$lt':end}}
    initt  = time.time()
    vis = list(visi_original_tb.find(where))
    
    print(len(vis))
    visinfos_zmd, visinfos_qs, visinfos_mg, visinfos_hd, visinfos_xy, visinfos_ls, visinfos_sj = [], [], [], [], [], [], []
    segments = [zmd, qs, mg, hd, xy, ls, sj]
    visinfos = [visinfos_zmd, visinfos_qs, visinfos_mg, visinfos_hd, visinfos_xy, visinfos_ls, visinfos_sj]
    for v in vis:
        cameraid = v['cameraid']
        for i, seg in enumerate(segments):
            if cameraid in seg:
                segment = i
                break
        visinfos[segment].append(v)

    names = ['驻马店', '确山', '明港', '胡店', '信阳', '灵山', '省界']
    viss = [(0, 0, 0) for i in range(7)]
    for i, visinfo in enumerate(visinfos):
        for info in visinfo:
            viss[i][0] += info['visibility']
            viss[i][1] += 1
        if viss[i][1]!= 0:
            viss[i][2] = viss[0][1]/viss[i][1]
            print(name[i], 't', str(viss[i][2]))

    print('cost time', str(time.time() - initt))

    time.sleep(30)
