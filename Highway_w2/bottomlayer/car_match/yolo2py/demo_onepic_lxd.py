#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:16:59 2018

@author: yan
"""

import os
import cv2
import numpy as np
from torch.multiprocessing import Pool

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
from PIL import Image
import ipdb

# This prevents deadlocks in the data loader, caused by
# some incompatibility between pytorch and cv2 multiprocessing.
# See https://github.com/pytorch/pytorch/issues/1355.
#cv2.setNumThreads(0)

def preprocess(fname):
    # return fname
    # read the pic  fname is the path of pic
#    ipdb.set_trace()
    image = cv2.imread(fname)
    im_data = np.expand_dims(
        yolo_utils.preprocess_test((image, None, cfg.multi_scale_inp_size), 0)[0], 0)
    return image, im_data


# hyper-parameters
# npz_fname = 'models/yolo-voc.weights.npz'
# h5_fname = 'models/yolo-voc.weights.h5'
trained_model = '../ten_thous_models/36_epoch.h5'
# trained_model = os.path.join(
#     cfg.train_output_dir, 'darknet19_voc07trainval_exp3_158.h5')
thresh = 0.5
#im_path = '../胡店段TV_scrhot'
#im_path = ''
im_path = '../../胡店段tv44_tv8new/TV35_20180711124236_20180711144236_78525916'

labels = ('car', 'truck', 'trailer', 'oil')
# ---

net = Darknet19()
net_utils.load_net(trained_model, net)
# net.load_from_npz(npz_fname)
# net_utils.save_net(h5_fname, net)
net.cuda()
net.eval()
print('load model succ...')

#t_det = Timer()
#t_total = Timer()
# =============================================================================
# im_fnames = sorted((fname
#                     for fname in os.listdir(im_path)
#                     if os.path.splitext(fname)[-1] == '.jpg'),key =lambda x:float(x.split('+')[1][:-4]))
# =============================================================================
im_fnames = sorted(fname for fname in os.listdir(im_path))

save_small_pic_name = im_fnames
num_save =  0
name_num = 0
jishu =0

TIME = "12.44.47"
h,m,s = TIME.strip().split(".")
s_dot = int(h) * 3600 + int(m) * 60 + int(s)
#im_fnames = os.listdir(im_path)
#im_fnames.sort(key = lambda x:float(x.split('+')[1][:-4]))
#print(im_fnames)
#im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)
#print(im_fnames)
#pool = Pool(processes=1)
print(trained_model)

for im_path_irt in im_fnames:
    print(im_path_irt)
    print(jishu)
    full_path =  os.path.join(im_path, im_path_irt)
#    ipdb.set_trace()
    

    image, im_data = preprocess(full_path)
    
    im_data = net_utils.np_to_variable(
        im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
    
    bbox_pred, iou_pred, prob_pred = net(im_data)
    
    bbox_pred = bbox_pred.data.cpu().numpy()
    iou_pred = iou_pred.data.cpu().numpy()
    prob_pred = prob_pred.data.cpu().numpy()
    
    bboxes, scores, cls_inds = yolo_utils.postprocess(
            bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh,size_index=0)
        
    
    for i_num, box in enumerate(bboxes):
#        print('start_save_crop')
#        if scores[i_num] < 0.65 :
#            name_num= name_num +1
#            continue
#        name_num = i_num
        cls_indx = cls_inds[i_num]
        if cls_indx == 0 :
            name_num= name_num +1
            continue
        imagepil = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        box_crop = (box[0],box[1],box[2],box[3])
        
        crop_image = imagepil.crop(box_crop)
        
        time_of_spicname_num = int(((int(im_path_irt.split('_')[0])-1)*0.4+s_dot)*10)
        time_n =8-len(str(time_of_spicname_num))
        time_of_spicname = str(0)*time_n + str(time_of_spicname_num)
        
        crop_image.save("../../胡店段tv44_tv8new/test_spic_512_6/"+time_of_spicname+'_'+str(i_num-name_num)+
                        '_'+labels[cls_indx]+'_'+
                        im_path_irt.split('_')[1]+'_'+str(box[0])+'_'
                        +str(box[1])+'_'+str(box[2])+'_'+str(box[3])+".jpg")
        
#        crop_image.save("../../胡店段tv44_tv8new/tv48_spic/"+labels[cls_indx]+
#                        str(i_num-name_num)+'+'+
#                        str(box[0])+'+'+str(box[1])+'+'+str(box[2])+'+'+str(box[3])+'+'+
#                        im_path_irt.split('.')[0]+'.'+im_path_irt.split('.')[1]+".jpg")
#        cv2.imwrite("../胡店段_spic"+labels[cls_indx]+str(i_num-name_num)+'+'+save_small_pic_name[i].split('.')[0]+".jpg",crop_image)
        print('save_com')
    name_num = 0
    
    jishu = jishu+1
     
     



#for i, (image, im_data) in enumerate(pool.imap(
#        preprocess, im_fnames, chunksize=1)):
#    t_total.tic()
#    im_data = net_utils.np_to_variable(
#        im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
#    t_det.tic()
#    bbox_pred, iou_pred, prob_pred = net(im_data)
#    det_time = t_det.toc()
##    ipdb.set_trace()
#    # to numpy
#    bbox_pred = bbox_pred.data.cpu().numpy()
#    iou_pred = iou_pred.data.cpu().numpy()
#    prob_pred = prob_pred.data.cpu().numpy()
#
#    # print bbox_pred.shape, iou_pred.shape, prob_pred.shape
#
#    bboxes, scores, cls_inds = yolo_utils.postprocess(
#        bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh)
#    
##    print('bbox_axis',bboxes)
##    print('scores',scores)
##    print('class',cls_inds)
#    for i_num, box in enumerate(bboxes):
##        print('start_save_crop')
#        if scores[i_num] < 0.65:
#            name_num= name_num +1
#            continue
##        name_num = i_num
#        cls_indx = cls_inds[i_num]
#        imagepil = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
#        box_crop = (box[0],box[1],box[2],box[3])
#        
#        crop_image = imagepil.crop(box_crop)
#        crop_image.save("../胡店段_tv44_spic/"+labels[cls_indx]+str(i_num-name_num)+'+'+save_small_pic_name[i].split('.')[0]+'.'+save_small_pic_name[i].split('.')[1]+".jpg")
##        cv2.imwrite("../胡店段_spic"+labels[cls_indx]+str(i_num-name_num)+'+'+save_small_pic_name[i].split('.')[0]+".jpg",crop_image)
#        print('save_com')
#    name_num = 0
##        image1 = image.crop(crop_box)
##        image1.save('image1.jpg')
#        
#
##        thick = int((h + w) / 300)
##        cv2.rectangle(imgcv,
##                      (box[0], box[1]), (box[2], box[3]),
##                      colors[cls_indx], thick)
##        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
##        cv2.putText(imgcv, mess, (box[0], box[1] - 12),
##                    0, 1e-3 * h, colors[cls_indx], thick // 3)
##    
#
#    im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)
#
#    if im2show.shape[0] > 1100:
#        im2show = cv2.resize(im2show,
#                             (int(1000. *
#                                  float(im2show.shape[1]) / im2show.shape[0]),
#                              1000))
#    cv2.imshow('test', im2show)
#
#    total_time = t_total.toc()
#    # wait_time = max(int(60 - total_time * 1000), 1)
#    cv2.waitKey(0)
#
#    if i % 1 == 0:
#        format_str = 'frame: %d, ' \
#                     '(detection: %.1f Hz, %.1f ms) ' \
#                     '(total: %.1f Hz, %.1f ms)'
#        print((format_str % (
#            i,
#            1. / det_time, det_time * 1000,
#            1. / total_time, total_time * 1000)))
#
#        t_total.clear()
#        t_det.clear()
