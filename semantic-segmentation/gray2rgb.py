import numpy as np
import os
from PIL import Image
import cv2

ignore_label=0
camvid_colors = [
                 [128, 128, 128],
                 [128,   0,   0],
                 [192, 192, 128],
                 [255,  69,   0],
                 [128,  64, 128],
                 [60,   40, 222],
                 [128, 128,   0],
                 [192, 128, 128],
                 [64,   64, 128],
                 [64,    0, 128],
                 [64,   64,  0 ],
                 [0,   128, 192],
                 [0,     0,   0]
                ]

city_colors = [
               [128,  64, 128],
               [244,  35, 232],
               [ 70,  70,  70],
               [102, 102, 156],
               [190, 153, 153],
               [153, 153, 153],
               [250, 170,  30],
               [220, 220,   0],
               [107, 142,  35],
               [152, 251, 152],
               [ 0, 130, 180],
               [220,  20,  60],
               [255,   0,   0],
               [  0,   0, 142],
               [  0,   0,  70],
               [  0,  60, 100],
               [  0,  80, 100],
               [  0,   0, 230],
               [119,  11,  32]
              ]


def trans_id(pred_label):
    ignore_label=250
    id_to_transform = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    mask=np.zeros((512,1024))
    for k,v in id_to_transform.items():
        mask[pred_label==k]=v
    return mask

def convert(path,num_class):
    imgname=os.listdir(path)
    for name in imgname:
        img=Image.open(path+name).convert('L')
        if not os.path.exists('./color'):
            os.makedirs('./color')
        arr=np.array(img)
        w,h=np.shape(arr)
        arr=trans_id(arr)
        output0=np.zeros((w,h))
        output1=np.zeros((w,h))
        output2=np.zeros((w,h))
        for label in range(0,num_class):
            mask=arr==label
            output0[mask]=city_colors[label][0]
            output1[mask]=city_colors[label][1]
            output2[mask]=city_colors[label][2]
        output0=output0[:,:,np.newaxis]
        output1=output1[:,:,np.newaxis]
        output2=output2[:,:,np.newaxis]
        #因为在opencv中图片格式为[h,w,c]，所以在第三个轴拼接
        output=np.concatenate((output2,output1,output0),axis=2)
        print(output)
        cv2.imwrite("./color/"+name,output)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./gray_results/', type=str)
    parser.add_argument('--num_class', default='19', type=int)
    args = parser.parse_args()
    path,num_class=args.path,args.num_class
    convert(path, num_class)

if __name__=="__main__":
    """
    python gray_rgb.py --path gray_results/ --num_class 19
    """
    main()
