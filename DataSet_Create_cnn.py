import tensorflow as tf
import numpy as np
import cv2
import os
import sys
from six.moves import cPickle

def main():
    pklPath = r"E:\Deep_learn\Cap_detection\DataSet1"+"\\test.pkl"
    input_size = (64, 64)
    data_oneDim = input_size[0] * input_size[1] * 3
    # 测试集图片的张数
    num_trainImage = 4000
    data = np.array([[0 for x in range(data_oneDim)] for y in range(num_trainImage)])
    # 就本数据集而言，二分类，正反样例各占50% ， 所以每种类设置数量为 num_trainImage/2 标签
    label1 = [0 for x in range(int(num_trainImage/2))]
    label2 = [1 for x in range(int(num_trainImage/2))]
    label = np.array(label1 + label2)


    # 上下各2000张 每张64*64  3通道
    # 64*64*3 = 12288   4000*12288 的 numpy的uint8s数组
    ImageFile1 = r"E:\Deep_learn\Cap_detection\DataSet1\Test\0" + "\\"
    ImageNames1 = os.listdir(ImageFile1)
    i = 0
    for Name1 in ImageNames1:
        imagePath = ImageFile1 + Name1
        src = cv2.imread(imagePath)
        src = cv2.resize(src, (64, 64))
        src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)  # 因为cv2.imread  opencv中默认颜色通道为BGR
        # src_data = np.array([src[:,:,0],src[:,:,1],src[:,:,2]])
        src_data = np.array(src)
        src_data = src_data.reshape(12288)
        data[i] = src_data
        i = i + 1
        # print(src[0,:,0])      测试输出  转化后的src_data 第一行的 r 分量是否 和src一致
        # print(src_data[:64])
        # print(src.shape)
        # print(src_data.shape)

    ImageFile2 = r"E:\Deep_learn\Cap_detection\DataSet1\Test\1" + "\\"
    ImageNames2 = os.listdir(ImageFile2)
    for Name2 in ImageNames2:
        imagePath = ImageFile2 + Name2
        src = cv2.imread(imagePath)
        src = cv2.resize(src, input_size)
        src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)  # 因为cv2.imread  opencv中默认颜色通道为BGR
        # src_data = np.array([src[:,:,0],src[:,:,1],src[:,:,2]])
        src_data = np.array(src)
        src_data = src_data.reshape(data_oneDim)
        data[i] = src_data
        i = i + 1

    Cap_dict = { 'data': data , 'labels':label }
    print(Cap_dict['data'].shape)

    with open(pklPath,'wb') as f:
        cPickle.dump(Cap_dict,f)

if __name__ == "__main__":
    main()

# imageFile = r"E:\Deep_learn\Cap_detection\DataSet\Test" + "\\"
# pklPath = r"E:\Deep_learn\Cap_detection\DataSet"+"\\test.pkl"
#
# data = np.array([[0 for x in range(12288)] for y in range(1552)])
# label1 = [0 for x in range(776)]
# label2 = [1 for x in range(776)]
# label = np.array(label1 + label2)
#
# # print(label[1],label[1999],label[2000],label[3999])
# # 上下各2000张 每张64*64  3通道
# # 64*64*3 = 12288   4000*12288 的 numpy的uint8s数组
# for i in range(1,777):
#     imagePath = imageFile + "0\\" + str(i) + ".jpg"
#     src = cv2.imread(imagePath)
#     src = cv2.resize(src, (64, 64))
#     src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # 因为cv2.imread  opencv中默认颜色通道为BGR
#     # src_data = np.array([src[:, :, 0], src[:, :, 1], src[:, :, 2]])
#     src_data = np.array(src)
#     src_data = src_data.reshape(12288)
#     data[i-1] = src_data
#     # print(src[0,:,0])      测试输出  转化后的src_data 第一行的 r 分量是否 和src一致
#     # print(src_data[:64])
#     # print(src.shape)
#     # print(src_data.shape)
# for i in range(777,1553):
#     imagePath = imageFile + "1\\" + str(i) + ".jpg"
#     src = cv2.imread(imagePath)
#     src = cv2.resize(src, (64, 64))
#     src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # 因为cv2.imread  opencv中默认颜色通道为BGR
#     # src_data = np.array([src[:, :, 0], src[:, :, 1],src[:, :, 2]])
#     src_data = np.array(src)
#     src_data = src_data.reshape(12288)
#     data[i-1] = src_data
#
# Cap_dict = { 'data': data , 'labels':label }
# print(Cap_dict['data'].shape)
#
# with open(pklPath, 'wb') as f:
#     cPickle.dump(Cap_dict, f)


