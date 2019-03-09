import pandas as pd
import os
import time
import numpy as np
from carmatch.geo_config import geo_distance, segments
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable



class BaseNetwork(nn.Module):

    def __init__(self, modelname):
        super(BaseNetwork, self).__init__()
        self.modelname = modelname
        if modelname == 'Vgg16':
            self.CNN = models.vgg16(pretrained=True).features
            self.FC1 = nn.Linear(7*7*512, 2048)
            self.FC2 = nn.Linear(2048, 128)
        else:
            raise ('Please select model')

    def forward(self, x):
        if self.modelname == 'Vgg16':
            output = self.CNN(x)
            output = output.view(output.size()[0], -1)
            output = self.FC1(output)
            output = F.relu(output)
            output = self.FC2(output)
        else:
            raise ('Please select model')
        return output

class similarity():
    def __init__(self):
        self.model = BaseNetwork('Vgg16')
        self.model.load_state_dict(torch.load(modelpath))
        self.model.cuda()
    	#self.model = torch.load(modelpath).cuda()
        self.transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #return feature by list
    def get_pil_feat(self, pilimage):
        starttime = time.time()
        tensorimage = Variable(self.transform(pilimage)).view(1, 3, 224, 224).cuda()
        feature = self.model(tensorimage)
        feature = list(feature.cpu().data.numpy().flatten()*1000)
        feature = [int(f) for f in feature]
        return feature

def cal_closestcars(feature, df):
    '''
    feature    108 vectors, to cal image distace
    df         pandas dataframe
    return     less than 3 closest cars
    '''
    df_tar = df.copy()
    df_tar['distance'] = df_tar['feature'].apply(
        lambda x:np.linalg.norm(np.array(x) - feature)
    )
    df_tar = df_tar.sort_values(by='distance')
    carsinfo = df_tar[['distance', 'carid', 'imagepath', 'time', 'cameraid', 'location', 'position']].values
    return carsinfo[:3]


modelpath = '../../model/CCL_epoch10_305000.pt'
class MatchCar():
    def __init__(self):
        self.similarity = similarity()
        self.cam_start = 36
        self.cam_end = 63
    
    def set_cardf(self, df):
        self.car_df = df

    def get_pil_feat(self, pilimage, t, cam_id):
        feat_anc = self.similarity.get_pil_feat(pilimage)
        return feat_anc, t, cam_id
   
    def spec_search(self, camid_anc, feat_anc, t_anc, camid_tar):
        '''
        give anchor car info, search three closest cars in self.car_df        

        camid_anc    anchor cameraid
        feat_anc     anchor car feature
        t_anc        anchor car capture time
        camid_tar   target search cameraid
        '''
        geo_tar = geo_distance[camid_tar]
        geo_anc = geo_distance[camid_anc]
        dis_rel = (geo_tar[0] - geo_anc[0])*1000 + (geo_tar[1]-geo_anc[1])
        
        if dis_rel > 0:
            tar_start_time = t_anc + dis_rel*3.6/150
            tar_end_time = t_anc + dis_rel*3.6/30
            if camid_tar == 'TV52': tar_end_time += 120
        elif dis_rel < 0:
            tar_start_time = t_anc + dis_rel*3.6/30
            tar_end_time = t_anc + dis_rel*3.6/150
        else:
            tar_start_time = t_anc -10
            tar_end_time = t_anc +10
        
        '''
        random_break = random.randint(0,100)
        if random_break>20:
            tar_df = self.car_df[(self.car_df['cameraid']==camid_tar) & (self.car_df['time']>tar_start_time) & (self.car_df['time']<tar_end_time)]
        else:
            print(camid_tar + '  breakdown')
            tar_df = self.car_df[(self.car_df['cameraid']=='TV_')]
        
        '''
        tar_df = self.car_df[(self.car_df['cameraid']==camid_tar) & (self.car_df['time']>tar_start_time) & (self.car_df['time']<tar_end_time)]
        
        closest_cars = cal_closestcars(feat_anc, tar_df)
        return closest_cars
    
    def get_ancbase_dis(self, anchor):
        pass

    def forward_search(self, camid_anc, feat_anc, t_anc):
        camid_cursor = camid_anc
        t_cursor = t_anc

        searchdone = False
        samecars_forward = []

        camnum_anc = int(camid_anc.split('V')[-1])
        search_cams = ['TV%d' %i for i in range(camnum_anc, self.cam_end+1)]
        for camid_tar in search_cams:
            ans_cars = self.spec_search(camid_cursor, feat_anc, t_cursor, camid_tar)
            
            for index, car in enumerate(ans_cars):
                if index == 0 and car[0] > 30000:
                    return samecars_forward
                elif index == 0 and car[0]<= 30000:
                    camid_cursor = camid_tar
                    t_cursor = car[3]
                
                if car[0] <= 30000:
                    samecars_forward.append(car)
        return samecars_forward

    def backward_search(self, camid_anc, feat_anc, t_anc):
        camid_cursor = camid_anc
        t_cursor = t_anc

        searchdone = False
        samecars_backward = []

        camnum_anc = int(camid_anc.split('V')[-1])
        search_cams = ['TV%d' %i for i in range(camnum_anc, self.cam_start-1, -1)]
        for camid_tar in search_cams:
            ans_cars = self.spec_search(camid_cursor, feat_anc, t_cursor, camid_tar)
            
            for index, car in enumerate(ans_cars):
                if index == 0 and car[0] > 30000:
                    return samecars_backward
                elif index == 0 and car[0]<= 30000:
                    camid_cursor = camid_tar
                    t_cursor = car[3]
                
                if car[0] <= 30000 and camid_tar != camid_anc:
                    samecars_backward.append(car)
        return samecars_backward

    def rough_search(self, camid_anc, feat_anc, t_anc):
        #self.disbase.....
        
        samecars_forward = self.forward_search(camid_anc, feat_anc, t_anc)
        samecars_backward = self.backward_search(camid_anc, feat_anc, t_anc)
        
        samecars_rough = samecars_forward + samecars_backward
        samecars_rough = sorted(samecars_rough, key=lambda x:[x[3]])
        
        
        cam_ids = list(set([info[4] for info in samecars_rough]))
        cam_ids.sort()
        print(cam_ids)

        return samecars_rough, cam_ids

    def get_direction(self, cars):
        if 'TV50' in cars and 'TV52' not in cars:
            return 'f'
        elif 'TV52' in cars and 'TV50' not in cars:
            return 'b'
        else:
            return ''

    def zone_search(self, samecars_rough, cam_ids, feat_anc, direction):
        if direction == 'f': 
            print('forward zone search')
            camid_anc = 'TV50'
            camid_sear = 'TV52'
        elif direction == 'b': 
            print('backward zone search')
            camid_anc = 'TV52'
            camid_sear = 'TV50'
        else:
            return '', 0
        
        tar_start_time = 0
        tar_end_time = 0
        num_zone = 0
        dis_base = 0
        tar_t = 0
        for samecar in samecars_rough:
            dis_base += samecar[0]
            if samecar[4] == camid_anc:
                tar_t += samecar[3]
                num_zone += 1
        dis_base = dis_base/len(samecars_rough)
        
        if direction == 'f':
            tar_start_time = tar_t/num_zone + 2500*3.6/150
            tar_end_time = tar_start_time + 1800
        else:
            tar_end_time = tar_t/num_zone - 2500*3.6/150
            tar_start_time = tar_end_time - 1800
        
        while True:
            tar_df = self.car_df[(self.car_df['cameraid'] == camid_sear) & (self.car_df['time'] > tar_start_time) & (self.car_df['time'] < tar_end_time)]
            closest_cars = cal_closestcars(feat_anc, tar_df)
            if not len(closest_cars):
                print('zone search end, not found')
                break
            
            if direction == 'f':
                tar_start_time = tar_end_time
                tar_end_time = tar_end_time + 1800
            elif direction == 'b':
                tar_end_time = tar_start_time
                tar_start_time = tar_end_time -1800
            else:
                print('zone search error')

            if closest_cars[0][0] < dis_base * 2:
                t_cursor = closest_cars[0][3]
                dis = closest_cars[0][0]
                ava_dis = dis
                ava_dis_count = 1
                id_cursor = 52 if (direction == 'f') else 50
                idlist = [52, 53] if (direction == 'f') else [49, 48]
                for i in idlist:
                    camid_tar = 'TV%d' %(i+1) if (direction == 'f') else 'TV%d' %(i-1)
                    samecars = self.spec_search('TV%d' %id_cursor, feat_anc, t_cursor, camid_tar)
                    if len(samecars) and (samecars[0][0] < 2 * ava_dis):
                        t_cursor = samecars[0][3]
                        dis += samecars[0][0]
                        ava_dis_count += 1
                        id_cursor = int(camid_tar.split('V')[-1])

                ava_dis = dis/ava_dis_count
                if ava_dis < dis_base * 1.5:
                    return direction, closest_cars[0][3]

        return '', 0
    
    def samecars_fix(self, direction, t, feat_anc):
        if not direction:
            return False, []
        elif direction == 'f':
            samecars = self.forward_search('TV52', feat_anc, t)
            return True, samecars
        elif direction == 'b':
            samecars = self.backward_search('TV50', feat_anc, t)
            return True, samecars

    def samecars_modify(self, fixed_cars, anchorid):
        segs = segments
        diss = [0, 0, 0, 0, 0]
        counts = [0, 0, 0, 0, 0]
        
        #find anchor segment
        anchorseg = 0
        for i, seg in enumerate(segs):
            if anchorid in seg:
                anchorseg = i
                break
        #get avadis for every segs
        for car in fixed_cars:
            for i in range(5):
                if car[4] in segs[i]:
                    diss[i] += car[0]
                    counts[i] += 1
                    break
        for i in range(5):
            if counts[i] == 0: diss[i] = float('Inf')
            else: diss[i] = diss[i]/counts[i]
        #final list
        final_list = []
        for i in range(5):
            if diss[i] < diss[anchorseg] * 1.75:
                final_list += segs[i]
        samecars = []
        cams = []
        for car in fixed_cars:
            if car[4] in final_list:
                samecars.append(car)
                cams.append(car[4])
        cams = list(set(cams))
        cams.sort()
        samecars = sorted(samecars, key=lambda x:[x[3]])
        return samecars, cams

    def searchcar(self, pilimage, t_anc, camid_anc, df):

        initt = time.time()
        self.set_cardf(df)
        feat_anc, t_anc, camid_anc = self.get_pil_feat(pilimage, t_anc, camid_anc)
        #print('anchor:\t%f\t%s'%(t_anc, camid_anc))
        answer, cameras = self.rough_search(camid_anc, feat_anc, t_anc)
        #print('rough_search one')
        cars, cam = self.samecars_modify(answer, camid_anc)
        #print('roughsearch_modify\t%s'%(str(cam)))
        direction = self.get_direction(cam)
        direction, t = self.zone_search(cars, cam, feat_anc, direction)
        #print('direction:\t',str(direction))
        ret, samecars = self.samecars_fix(direction, t, feat_anc)
        answer = cars + samecars
        #print('samecars_fixed')
        final_cars, cam = self.samecars_modify(answer, camid_anc)
        #print('search method cost\t', str(time.time() - initt))
        
        return final_cars
        
    
    def download(self, samecar_rough, num):
        if not os.path.exists('/media/assests/Test/%d' %num):
                os.mkdir('/media/assests/Test/%d' %num)
        for i, carinfo in enumerate(samecar_rough):
            imagepath = 'http://192.168.5.41:8080/assests' + carinfo[2].split('assests')[-1]
            img = urlimread(imagepath)

            cameraid = carinfo[2].split('/')[-2]
            name = cameraid + '_' + carinfo[2].split('/')[-1]
            io.imsave('/media/assests/Test/%d/%s' %(num, name),img)

if __name__ == '__main__':
    car_tb = MongoClient('mongodb://192.168.5.41:27017')['highway']['carmatch_2018-7-14']
    initt = time.time()
    df = pd.DataFrame(list(car_tb.find()))

    li = [11517, 15007, 19544, 24430, 26175, 26524, 26873, 27920, 28269,28618,29665, 30014, 31410, 32806, 33155, 34202, 34900]
    #li = [11517]
    for i in li:
        initt = time.time()
        print('\n\n------%d------' %i)
        anchor = df.iloc[i]
        feature = anchor['feature']
        anchort = anchor['time']
        anchorid = anchor['cameraid']
        print('anchor:\t%f\t%s'%(anchort, anchorid))

        match = MatchCar()
        match.set_cardf(df)
        answer, cameras = match.rough_search(anchorid, feature, anchort)
        print('rough_search one')
        cars, cam = match.samecars_modify(answer, anchorid)
        print('roughsearch_modify\t%s'%(str(cam)))
        direction = match.get_direction(cam)
        direction, t = match.zone_search(cars, cam, feature, direction)
        print('direction:\t',str(direction))
        ret, samecars = match.samecars_fix(direction, t, feature)
        answer = cars + samecars
        print('samecars_fixed')
        final_cars, cam = match.samecars_modify(answer, anchorid)
        print('dt', str(time.time() - initt))
        match.download(final_cars, i)
