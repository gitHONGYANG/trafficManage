import scipy.ndimage as ndimage
import numpy as np
from numba import jit
import cv2

@jit
def calc_variance(img, size=5, stride=2):
    """
    caculating variance of input image
    """
    # element-wise sqaure of image, cause D(x) = E(x^2) - E(x)*E(x)
    img_square = np.square(img) 
    
    # calculating convolution 
    conv = ndimage.convolve
    kernel = np.zeros((size, size)) + 1
    kernel_sum = kernel.shape[0]*kernel.shape[1]

    # calculating E(x)
    E_img = conv(img, kernel, mode='constant')[::stride, ::stride]/kernel_sum

    # caculating E(x^2)
    E_img_square = conv(img_square, kernel, mode='constant')[::stride, ::stride]/kernel_sum

    # D(x) = E(x^2) - E(x)*E(x)
    variance = E_img_square - np.square(E_img)

    # minmax normalization to 0-1
    variance = (variance-variance.min())/(variance.max()-variance.min())

    # the range is in [0, 1]
    return variance 

'''
形态学运算
输入
    open_kernel, close_kernel:开闭运算核
输出
    img:cv二值化图像,open_kernel,close_kernel:卷积核
'''
def morph(img, open_kernel, close_kernel):
    morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel)
    morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
    #这里需要改进
    open0 = cv2.morphologyEx(img, cv2.MORPH_OPEN, morph_open_kernel)
    open1 = cv2.morphologyEx(open0, cv2.MORPH_OPEN, morph_open_kernel)
    close0 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, morph_close_kernel)

    return cv2.morphologyEx(close0, cv2.MORPH_CLOSE, morph_close_kernel)

'''
密度筛选，面积筛选
输入
    cnts:轮廓, area_threshold_min:int, area_threshold_max:int,height:原始图像高度
输出
    cnt_pos_size:筛选后轮廓信息  [[x+int(w/2), y+int(h/2), w, h, 出现时间0, 0]]
    筛选后扔掉部分
'''
def contour_density_filter(cnts, area_threshold_min, area_threshold_max, height):
    cnt_pos_size = []
    cnt_filtered= []
    #filter contours by area get every contour's info
    for temp in cnts:
        (x, y, w, h) = cv2.boundingRect(temp)
        area1 = area_threshold_min
        area2 = area_threshold_max * (y/(2*height) + 0.5)
        area_temp = cv2.contourArea(temp)
        if area_temp > area1 and area_temp < area2:
            (x, y, w, h) = cv2.boundingRect(temp)
            cnt_filtered.append(temp)
    #[0]:center x, [1]:center y, [2]:width, [3]:height, [4]:occ num ,[5]:fod_id
            cnt_pos_size.append([x+int(w/2), y+int(h/2), w, h, 0, 0])
    length = len(cnt_pos_size)
    del_flag = np.zeros([length+1, length])
    for i in range(length):
        for temp in range(length):
            if (abs(cnt_pos_size[i][0] - cnt_pos_size[temp][0]) < 80) & (abs(cnt_pos_size[i][1] - cnt_pos_size[temp][1]) < 100):
                #if cv2.contourArea(cnt_pos_size(temp))
                del_flag[i][temp] = 1
        if cnt_pos_size[i][2] < 10 or cnt_pos_size[i][3] < 10:
            del_flag[i][i] = 6
        if del_flag[i].sum() < 6:
            del_flag[i] = 0
        else:
            del_flag[i][i] = 1
    templist = cnt_pos_size[:]
    cnt_del = []
    for i in range(length):
        if(del_flag[:-2, i].sum() > 0):
            cnt_pos_size.remove(templist[i])
            cnt_del.append(i)
    del templist[:]
    cnt_filtered = np.delete(cnt_filtered, cnt_del, 0)
    return cnt_pos_size, cnt_filtered


def contour_density_filter2(cnts, area_threshold_min, area_threshold_max, height):
    cnt_pos_size = []
    cnt_filtered= []
    #filter contours by area get every contour's info
    for temp in cnts:
        (x, y, w, h) = cv2.boundingRect(temp)
        area1 = area_threshold_min
        area2 = area_threshold_max * (y/(2*height) + 0.5)
        area_temp = cv2.contourArea(temp)
        if area_temp > area1 and area_temp < area2:
            (x, y, w, h) = cv2.boundingRect(temp)
            cnt_filtered.append(temp)
    #[0]:center x, [1]:center y, [2]:width, [3]:height, [4]:occ num ,[5]:fod_id
            cnt_pos_size.append([x+int(w/2), y+int(h/2), w, h, 0, 0])

    return cnt_pos_size, cnt_filtered
'''
对原始掩模处理
输入:
    mask_org:掩模图片
输出:
    mask:处理后掩模
    mask_w,mask_h,roi_h,roi_w:掩模信息
'''
def extra_mask_roi(mask_org):
    mask_gray = cv2.cvtColor(mask_org, cv2.COLOR_BGR2GRAY)
    mask_h = mask_gray.shape[0]
    mask_w = mask_gray.shape[1]
    #get the road ROI from the road mask
    roi_w = 0
    roi_h = 0
    #weight
    for roi_w in range(mask_gray.shape[1]):
        if mask_gray[:, roi_w].sum() > 0:
            break
    #hight
    for roi_h in range(mask_gray.shape[0]):
        if mask_gray[roi_h, :].sum() > 0:
            break
    mask = (mask_org[roi_h:mask_h, roi_w:mask_w] == 255)
    return mask, mask_w, mask_h, roi_w, roi_h


if __name__=='__main__':
    from skimage.io import imread, imsave
    from skimage.color import rgb2gray
    img = imread('tv35.jpg')

    gray = rgb2gray(img)
    gray = gray.astype('float32')/255.0

    var = calc_variance(gray)
    imsave('rst.png', var)



