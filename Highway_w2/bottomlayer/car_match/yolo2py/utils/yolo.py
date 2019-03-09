import cv2
import numpy as np
from .im_transform import imcv2_affine_trans, imcv2_recolor
# from box import BoundBox, box_iou, prob_compare
from ..utils.nms_wrapper import nms
from ..utils.cython_yolo import yolo_to_bbox


# This prevents deadlocks in the data loader, caused by
# some incompatibility between pytorch and cv2 multiprocessing.
# See https://github.com/pytorch/pytorch/issues/1355.
cv2.setNumThreads(0)


def clip_boxes(boxes, im_shape):
    '''

    :param boxes:车在图像的位置 [xx,4]
    :param im_shape: 图像规定的大小 [2]
    :return: 规范后的boxes
    功能：Clip boxes to image boundaries.
    '''


    if boxes.shape[0] == 0:
        return boxes
    #由于boxes的第二个维度的大小就是4，所以boxes[:,0::4] 和 boxes[:,0]等价
    #把坐标限定范围
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)  #减一的意义在于？？
    #这里好像没有判断x1,x2;y1,y2的大小关系？？
    return boxes


def nms_detections(pred_boxes, scores, nms_thresh):
    '''

    :param pred_boxes:
    :param scores:
    :param nms_thresh:
    :return:
    '''
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)  #矩阵拼接
    keep = nms(dets, nms_thresh)    #from ..utils.nms_wrapper import nms   keep是转换后的dets与nums_thresh的组合
    return keep


def _offset_boxes(boxes, im_shape, scale, offs, flip):
    '''

    :param boxes:  原位置
    :param im_shape: 图像给定大小
    :param scale: 缩放幅度
    :param offs: 偏置
    :param flip: 是否翻转
    :return:
    坐标的重置处理
    '''
    if len(boxes) == 0:
        return boxes
    boxes = np.asarray(boxes, dtype=np.float)
    boxes *= scale
    boxes[:, 0::2] -= offs[0]
    boxes[:, 1::2] -= offs[1]  #x,y坐标同时减去一个偏置
    boxes = clip_boxes(boxes, im_shape)

    if flip:     #把图像翻转
        boxes_x = np.copy(boxes[:, 0])
        boxes[:, 0] = im_shape[1] - boxes[:, 2]
        boxes[:, 2] = im_shape[1] - boxes_x

    return boxes


def preprocess_train(data, size_index):
    '''
    参数情形与preprocess_test基本类似
    :param data:
    :param size_index:
    :return:转换后的图像im，原图像ori_im 修改后的boxes
    '''
    im_path, blob, inp_size = data          #blob是数据库变量

    boxes, gt_classes = blob['boxes'], blob['gt_classes']

    im = cv2.imread(im_path)
    ori_im = np.copy(im)

    im, trans_param = imcv2_affine_trans(im)       #图像转换
    scale, offs, flip = trans_param
    boxes = _offset_boxes(boxes, im.shape, scale, offs, flip)

    if inp_size is not None and size_index is not None:
        inp_size = inp_size[size_index]
        w, h = inp_size
        boxes[:, 0::2] *= float(w) / im.shape[1]
        boxes[:, 1::2] *= float(h) / im.shape[0]     #resize
        im = cv2.resize(im, (w, h))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = imcv2_recolor(im)
    # im /= 255.

    # im = imcv2_recolor(im)
    # h, w = inp_size
    # im = cv2.resize(im, (w, h))
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im /= 255
    boxes = np.asarray(boxes, dtype=np.int)
    return im, boxes, gt_classes, [], ori_im


def preprocess_test(data, size_index):
    '''

    :param data: tulple类型 原输入：（image, None, cfg.multi_scale_inp_size)
    cfg.multi_scale_inp_size = [np.array([320, 320], dtype=np.int),
                        np.array([352, 352], dtype=np.int),
                        np.array([384, 384], dtype=np.int),
                        np.array([416, 416], dtype=np.int),
                        np.array([448, 448], dtype=np.int),
                        np.array([480, 480], dtype=np.int),
                        np.array([512, 512], dtype=np.int),
                        np.array([544, 544], dtype=np.int),
                        np.array([576, 576], dtype=np.int),
                        # np.array([608, 608], dtype=np.int),
                        ]
    :param size_index: 对multi_scale_inp_size提供的坐标大小限制进行选择
    :return: 转换后的图像im，原图像ori_im
    '''

    im, _, inp_size = data

    if isinstance(im, str):
        im = cv2.imread(im)
    ori_im = np.copy(im)

    #defalt w, h = 320, 320
    if inp_size is not None and size_index is not None:
        inp_size = inp_size[size_index]
        w, h = inp_size
        im = cv2.resize(im, (w, h))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im / 255.    #归一化

    return im, [], [], [], ori_im


def postprocess(bbox_pred, iou_pred, prob_pred, im_shape, cfg, thresh=0.05,
                size_index=0):
    """
    thresh: 阈值
    bbox_pred: (bsize, HxW, num_anchors, 4)
               ndarray of float (sig(tx), sig(ty), exp(tw), exp(th))
    iou_pred: (bsize, HxW, num_anchors, 1)   位置可信度
    prob_pred: (bsize, HxW, num_anchors, num_classes)  种类可信度
    cfg:cfg is from the cfgs/config.py and cfgs/config_voc.py
    before size_index =0 now change by ryan is 8 the fun call it default is 0
    """

    # num_classes, num_anchors = cfg.num_classes, cfg.num_anchors
    num_classes = cfg.num_classes   #num_class = 4
                                    #anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
                                                          #(6.63, 11.38), (9.42, 5.11), (16.62, 10.52)],
                                                          #dtype=np.float)
                                      #num_anchors = len(anchors)

    anchors = cfg.anchors
    W, H = cfg.multi_scale_out_size[size_index]
    assert bbox_pred.shape[0] == 1, 'postprocess only support one image per batch'  # noqa    断言语句

    bbox_pred = yolo_to_bbox(
        np.ascontiguousarray(bbox_pred, dtype=np.float),
        np.ascontiguousarray(anchors, dtype=np.float),     #复制bbox_pred，anchor数组 ，相当于copy()
        H, W)
    bbox_pred = np.reshape(bbox_pred, [-1, 4])  ####这里可设断点观察bbox_pred的大小情况，以及其值
    bbox_pred[:, 0::2] *= float(im_shape[1])
    bbox_pred[:, 1::2] *= float(im_shape[0])
    bbox_pred = bbox_pred.astype(np.int)

    iou_pred = np.reshape(iou_pred, [-1])#转换为一维数组
    prob_pred = np.reshape(prob_pred, [-1, num_classes])   #size  xx * 4  ####prob_preb的结构

    cls_inds = np.argmax(prob_pred, axis=1)   #确定网络探测的物品的种类
    prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]   #将相应的概率提取
    scores = iou_pred * prob_pred             #得到置信分数
    # scores = iou_pred
    assert len(scores) == len(bbox_pred), '{}, {}'.format(scores.shape, bbox_pred.shape)   #score与bbox_preb的形状必须一致
    # threshold
    keep = np.where(scores >= thresh)   #得到满足要求的坐标
    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]    #保留种类信息

    # NMS
    keep = np.zeros(len(bbox_pred), dtype=np.int)
    for i in range(num_classes):
        inds = np.where(cls_inds == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bbox_pred[inds]
        c_scores = scores[inds]     #把特定的种类提取出来
        c_keep = nms_detections(c_bboxes, c_scores, 0.3)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    # keep = nms_detections(bbox_pred, scores, 0.3)
    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]       #进一步提取

    # clip
    bbox_pred = clip_boxes(bbox_pred, im_shape)    #对坐标进行限制

    return bbox_pred, scores, cls_inds


def _bbox_targets_perimage(im_shape, gt_boxes, cls_inds, dontcare_areas, cfg):
    '''

    :param im_shape:图形大小
    :param gt_boxes:各个目标的位置
    :param cls_inds:各个目标的种类
    :param dontcare_areas:无
    :param cfg:配置
    :return:
    功能：
    '''
    # num_classes, num_anchors = cfg.num_classes, cfg.num_anchors
    # anchors = cfg.anchors
    H, W = cfg.out_size     #inp_size = np.array([416, 416], dtype=np.int)   # w, h
                            #out_size = inp_size / 32   H = 13， W = 13
    gt_boxes = np.asarray(gt_boxes, dtype=np.float)
    # TODO: dontcare areas
    dontcare_areas = np.asarray(dontcare_areas, dtype=np.float)

    # locate the cell of each gt_boxe
    cell_w = float(im_shape[1]) / W
    cell_h = float(im_shape[0]) / H
    cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5 / cell_w
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5 / cell_h   #中点坐标
    cell_inds = np.floor(cy) * W + np.floor(cx)           #向下取整
    cell_inds = cell_inds.astype(np.int)    ####cell_ind的各个值？？

    # [x1, y1, x2, y2],  [class]
    # gt_boxes[:, 0::2] /= im_shape[1]
    # gt_boxes[:, 1::2] /= im_shape[0]
    # gt_boxes[:, 0] = cx - np.floor(cx)
    # gt_boxes[:, 1] = cy - np.floor(cy)
    # gt_boxes[:, 2] = (gt_boxes[:, 2] - gt_boxes[:, 0]) / im_shape[1]
    # gt_boxes[:, 3] = (gt_boxes[:, 3] - gt_boxes[:, 1]) / im_shape[0]

    bbox_target = [[] for _ in range(H*W)]
    cls_target = [[] for _ in range(H*W)]  #这是为了方便维度的统一
    for i, ind in enumerate(cell_inds):
        bbox_target[ind].append(gt_boxes[i])
        cls_target[ind].append(cls_inds[i])   ####ind  ？？？
    return bbox_target, cls_target


def get_bbox_targets(images, gt_boxes, cls_inds, dontcares, cfg):
    '''
    :param images:
    :param gt_boxes:
    :param cls_inds:
    :param dontcares:  没有太大作用
    :param cfg:
    :return:
    功能：返回训练的target
    '''
    bbox_targets = []
    cls_targets = []
    for i, im in enumerate(images):
        bbox_target, cls_target = _bbox_targets_perimage(im.shape,
                                                         gt_boxes[i],
                                                         cls_inds[i],
                                                         dontcares[i],
                                                         cfg)
        bbox_targets.append(bbox_target)
        cls_targets.append(cls_target)
    return bbox_targets, cls_targets


def draw_detection(im, bboxes, scores, cls_inds, cfg, thr=0.3):
    '''

    :param im:
    :param bboxes:
    :param scores:
    :param cls_inds:
    :param cfg:
    :param thr:   阈值
    :return:
    功能：将图像目标的位置框出来
    '''
    # draw image
    colors = cfg.colors     #根据种类不同框的颜色不同
    labels = cfg.label_names   #四个种类

    imgcv = np.copy(im)   #不损伤原图
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)  #线的宽度
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 12),
                    0, 1e-3 * h, colors[cls_indx], thick // 3)  #将标签打上去

    return imgcv
