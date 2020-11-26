# real-time-segmentation
1.This codes is related to real-time semantic segmentaion which run on the datasets of cityscapes,camvid and pascal voc2012.

Now,the codes can only work on the cityscapes,because I have not complete the codes about camvid and pascal voc2012 yet,but the work is still go on.

2.This repository contain FPN_ASPP,Nonlocal_Network,and BiseNetv1,and I will build BiseNetv2 in the future.


3.The command to train the code is:

python main.py --cuda --model "choicemodel" train --datadir data  --num-epochs 30 --num-workers 4 --batch-size 4 --num-classes 20

# dependency
pytorch>=0.4.0

# models
1.FPN_ASPP 
2.Nonlocal_Network
3.BiseNet(simple version)

# result
![image](https://github.com/mshmoon/Real-Time-Segmentation/blob/master/semantic-segmentation/showresult/1.png)
![image](https://github.com/mshmoon/Real-Time-Segmentation/blob/master/semantic-segmentation/showresult/2.png)
