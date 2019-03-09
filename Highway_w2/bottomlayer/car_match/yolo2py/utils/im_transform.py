#本脚本的作用是对图像进行一定的处理
import numpy as np
import cv2


def imcv2_recolor(im, a=.1):   #im的维度一般是
    '''

    :param im:输入图像 必须是3通道 [xx,xx,3]
    :param a:  图像颜色处理的幅度，a的取值范围应在0到1之间
    :return: 转换后的图像
    功能：
    '''
    # t = [np.random.uniform()]
    # t += [np.random.uniform()]
    # t += [np.random.uniform()]
    # t = np.array(t) * 2. - 1.
    t = np.random.uniform(-1, 1, 3)   #在-1与1之间采3个点

    # random amplify each channel
    im = im.astype(np.float)
    im *= (1 + t * a)     #对图像的每个像素点进行处理 ，外加噪声？？？
    mx = 255. * (1 + a)
    up = np.random.uniform(-1, 1)
    im = np.power(im / mx, 1. + up * .5)
    # return np.array(im * 255., np.uint8)
    return im


def imcv2_affine_trans(im):
    '''
    :param im:  输入图像 3通道
    :return: 转换后的图像im ，截取区域起始点[offx, offy]，图像是否翻转
    功能：
    '''
    # Scale and translate
    h, w, c = im.shape
    scale = np.random.uniform() / 10. + 1.  #（1到1.1之间的随机数）
    max_offx = (scale - 1.) * w
    max_offy = (scale - 1.) * h    #确定边界值
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)#@param dsize output image size; if it equals zero, it is computed as:
    # .  \f[\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\f]
    #.   Either dsize or both fx and fy must be non-zero.
    im = im[offy: (offy + h), offx: (offx + w)]
    flip = np.random.uniform() > 0.5
    if flip:
        im = cv2.flip(im, 1)

    return im, [scale, [offx, offy], flip]
