# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .nms.cpu_nms import cpu_nms
from .nms.gpu_nms import gpu_nms    #这两个类的含义？？


# def nms(dets, thresh, force_cpu=False):
#     """Dispatch to either CPU or GPU NMS implementations."""
#
#     if dets.shape[0] == 0:
#         return []
#     if cfg.USE_GPU_NMS and not force_cpu:
#         return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
#     else:
#         return cpu_nms(dets, thresh)


def nms(dets, thresh, force_cpu=False):
    '''

    :param dets:
    :param thresh:
    :param force_cpu:  是否必须使用cpu
    :return:
    Dispatch to either CPU or GPU NMS implementations.
    '''


    if dets.shape[0] == 0:
        return []
    if force_cpu:
        return cpu_nms(dets, thresh)
    return gpu_nms(dets, thresh)
