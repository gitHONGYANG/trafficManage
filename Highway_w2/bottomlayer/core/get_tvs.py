from config import tvconfig_tb

active_tvs = list(tvconfig_tb.find({'name':'w2'}))[0]['tvs'] + list(tvconfig_tb.find({'name':'w3'}))[0]['tvs'] + list(tvconfig_tb.find({'name':'w4'}))[0]['tvs'] + list(tvconfig_tb.find({'name':'w5'}))[0]['tvs']
active_tvs.sort(key = lambda x: int(x.split('V')[-1]))
print(active_tvs)

info = list(tvconfig_tb.find({'name':'w2'}))[0]
tvs = info['tvs']
removelist = ['TV17', 'TV36', 'TV52', 'TV79']
for tv in removelist:
    try:
        tvs.remove(tv)
    except:
        pass
if 17>= int(active_tvs[0].split('V')[-1]):
    print('add TV17')
    tvs.append('TV17')


with open ('/home/highway/Highway/bottomlayer/core/tvs.txt', 'w') as f:
    for tv in tvs:
        f.writelines(tv+'\n')
