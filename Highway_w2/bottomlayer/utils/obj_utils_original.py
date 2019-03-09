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
    img_square = np.square(img)     #对图像像素平方操作
    
    # calculating convolution 
    conv = ndimage.convolve
    kernel = np.zeros((size, size)) + 1    #等价于np.ones((5,5))
    kernel_sum = kernel.shape[0]*kernel.shape[1]   #卷积核的大小

    # calculating E(x)
    E_img = conv(img, kernel, mode='constant')[::stride, ::stride]/kernel_sum  #对图像进行卷积操作

    # caculating E(x^2)
    E_img_square = conv(img_square, kernel, mode='constant')[::stride, ::stride]/kernel_sum

    # D(x) = E(x^2) - E(x)*E(x)
    variance = E_img_square - np.square(E_img)

    # minmax normalization to 0-1
    variance = (variance-variance.min())/(variance.max()-variance.min())

    # the range is in [0, 1]
    return variance 


def morph(img, open_kernel, close_kernel):
    '''
    :param img:  待处理图像
    :param open_kernel:
    :param close_kernel:  开闭运算核
    :return:
    功能：形态学运算

    图像形态学操作是  基于形状的一系列图像处理操作的合集，主要是基于集合论基础上的形态学数学。
    形态学有四个基本操作：膨胀、腐蚀、开、闭。
    膨胀与腐蚀是图像处理中最常用的形态学操作手段,常常被组合起来一起使用实现一些复杂的图像形态学操作。
    '''
    morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel,open_kernel))  #开运算的具体实现：通过先
    # 进行腐蚀操作，再进行膨胀操作得到。我们在移除小的对象时候很有用(假设物品是亮色，前景色是黑色)，被用来去除噪声。
    morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel,close_kernel))#闭运算是开运算的一个相反
    # 的操作，具体是先进行膨胀然后进行腐蚀操作。通常是被用来填充前景物体中的小洞
    open0 = cv2.morphologyEx(img, cv2.MORPH_OPEN, morph_open_kernel)
    open1 = cv2.morphologyEx(open0, cv2.MORPH_OPEN, morph_open_kernel)
    open1 = cv2.medianBlur(open1, close_kernel - open_kernel)
    of = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, morph_close_kernel)
    return of


def contour_density_filter(cnts, area_threshold_min, area_threshold_max, height):
    '''
    密度筛选，面积筛选
    输入
        cnts:轮廓，注意有一组的图片, area_threshold_min:int, area_threshold_max:int,height:原始图像高度
    输出
        cnt_pos_size:筛选后轮廓信息  [[x+int(w/2), y+int(h/2), w, h, 出现时间0, 0]]
        筛选后扔掉部分
    '''
    cnt_pos_size = []
    cnt_filtered= []
    #filter contours by area get every contour's info
    for temp in cnts:
        (x, y, w, h) = cv2.boundingRect(temp)  #计算轮廓的垂直边界最小矩形
        area1 = area_threshold_min
        area2 = area_threshold_max * (y/(2*height) + 0.5)  #  1/2*（ y + height）/ height
        area_temp = cv2.contourArea(temp)          # cvContourArea( contour,slice=CV_WHOLE_SEQ );
                                                   #     contour：轮廓（顶点的序列或数组）。
                                                   #     slice：感兴趣区轮廓部分的起点和终点，默认计算整个轮廓的面积。
                                                   #     函数cvContourArea计算整个或部分轮廓的面积。在计算部分轮廓的情况时，由轮廓弧线和连接两端点的弦
                                                   #     围成的区域总面积被计算
        if area_temp > area1 and area_temp < area2: #在限制范围内
            cnt_filtered.append(temp)
    #[0]:center x, [1]:center y, [2]:width, [3]:height, [4]:occ num ,[5]:fod_id
            cnt_pos_size.append([x+int(w/2), y+int(h/2), w, h, 0, 0])
    length = len(cnt_pos_size)    #得到筛选后的长度
    del_flag = np.zeros([length+1, length])
    for i in range(length):
        for temp in range(length):
            if (abs(cnt_pos_size[i][0] - cnt_pos_size[temp][0]) < 80) & (abs(cnt_pos_size[i][1] - cnt_pos_size[temp][1]) < 100):  #两个区域的中心坐标过近
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
    '''

    :param mask_org: 掩码图
    :return:
    '''
    mask_gray = cv2.cvtColor(mask_org, cv2.COLOR_BGR2GRAY) #转成灰度图
    mask_h = mask_gray.shape[0]
    mask_w = mask_gray.shape[1]   #图像高和宽
    #get the road ROI from the road mask
    #ROI（region of interest）  感兴趣区域
    roi_w = 0
    roi_h = 0
    #weight
    for roi_w in range(mask_gray.shape[1]):  #从0 到 w 一个个的测试
        if mask_gray[:, roi_w].sum() > 0:    #有非黑区域
            break
    #hight
    for roi_h in range(mask_gray.shape[0]):  #同宽度检测
        if mask_gray[roi_h, :].sum() > 0:
            break
    mask = (mask_org[roi_h:mask_h, roi_w:mask_w] == 255)   #将所得的正方区域颜色全部致白
    return mask, mask_w, mask_h, roi_w, roi_h


if __name__=='__main__':
    from skimage.io import imread, imsave
    from skimage.color import rgb2gray
    img = imread('tv35.jpg')

    gray = rgb2gray(img)
    gray = gray.astype('float32')/255.0

    var = calc_variance(gray)
    imsave('rst.png', var)



