import torch
import torch.nn as nn
import torch.nn.functional as F
from piwise import  model

class FPN_ASPP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = model.resnet18(pretrained=True)
        self.enc4_1=nn.Sequential(
            nn.Conv2d(512,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc3_1=nn.Sequential(
            nn.Conv2d(256,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc2_1=nn.Sequential(
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc1_1=nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.enc_1=nn.Sequential(
            nn.Conv2d(128,128,1,1,0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc_2=nn.Sequential(
            nn.Conv2d(128,128,1,1,0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc_3=nn.Sequential(
            nn.Conv2d(128,128,1,1,0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.res1=nn.Sequential(
            nn.Conv2d(128,64,1,1,0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,1,1,0),
            nn.BatchNorm2d(128),
        )
        self.conv_dilation_0=nn.Sequential(
            nn.Conv2d(128,128,1,1,0,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,1,1,0,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv_dilation_3=nn.Sequential(
            nn.Conv2d(128,128,3,1,3,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,3,3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv_dilation_6=nn.Sequential(
            nn.Conv2d(128,128,3,1,6,6),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,6,6),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.cut_dim=nn.Sequential(
            nn.Conv2d(128*3,128,1,1,0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU(inplace=True)
        self.classifer=nn.Sequential(
            nn.Conv2d(128, num_classes,1,1,0),
        )
       
    def forward(self,x):
     
        x4,x3,x2,x1,x0=self.model(x)

        a=self.enc4_1(x4)
        b=self.enc3_1(x3)
        c=self.enc2_1(x2)
        d=self.enc1_1(x1)

        b=self.sigmoid(F.upsample_bilinear(a,scale_factor=2))*self.enc_1(b)+F.upsample_bilinear(a,scale_factor=2)
        c=self.sigmoid(F.upsample_bilinear(b,scale_factor=2))*self.enc_2(c)+F.upsample_bilinear(b,scale_factor=2)
        d=self.sigmoid(F.upsample_bilinear(c,scale_factor=2))*self.enc_3(d)+F.upsample_bilinear(c,scale_factor=2)
        
        d=self.relu(self.res1(d)+d)

        d_0=self.conv_dilation_0(d)
        d_3=self.conv_dilation_3(d)
        d_6=self.conv_dilation_6(d)

        d=torch.cat([d_0,d_3,d_6],1)
        d=self.cut_dim(d)

        score=self.classifer(d)
        score=F.upsample_bilinear(score,scale_factor=4)

        return score


class Nonlocal_Network(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = model.resnet18(pretrained=True)

        self.enc4_1 = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc3_1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc2_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc1_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.enc_1 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc_2 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc_3 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1, 1, 0),
            nn.BatchNorm2d(128),

        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.bn1=nn.BatchNorm2d(128)
        self.classifer=nn.Conv2d(128,num_classes,1,1,0)

    def forward(self, x):
        x4, x3, x2, x1, x0 = self.model(x)

        a = self.enc4_1(x4)
        b = self.enc3_1(x3)
        c = self.enc2_1(x2)
        d = self.enc1_1(x1)

        b = self.sigmoid(F.upsample_bilinear(a, scale_factor=2)) * self.enc_1(b) + F.upsample_bilinear(a,
                                                                                                       scale_factor=2)
        c = self.sigmoid(F.upsample_bilinear(b, scale_factor=2)) * self.enc_2(c) + F.upsample_bilinear(b,
                                                                                                       scale_factor=2)
        d = self.sigmoid(F.upsample_bilinear(c, scale_factor=2)) * self.enc_3(d) + F.upsample_bilinear(c,
                                                                                                       scale_factor=2)

        d = self.relu(self.res1(d) + d)

        d = self.bn1(d * (1 + F.tanh(torch.matmul(d, d.transpose(3, 2)) @ d)))

        score = self.classifer(d)
        score = F.upsample_bilinear(score, scale_factor=4)

        return score


class BiseNetv1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet=model.resnet18(pretrained=True)
        self.Spatial_model = [self.resnet.layer1,self.resnet.layer2,self.resnet.layer3]
        self.Context_model = model.resnet18(pretrained=True)
        self.Spatial_entry=nn.Sequential(
            nn.Conv2d(3,64,1,2,0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )
        self.ARM_layer1=nn.Sequential(
            nn.Conv2d(128,128,1,1,0),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
        )
        self.ARM_layer2 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
        )
        self.FFM_layer1=nn.Sequential(
            nn.Conv2d(256,128,1,1,0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.FFM_layer2 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(512, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc2=nn.Sequential(
            nn.Conv2d(256,128,1,1,0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.classifer=nn.Conv2d(128,num_classes,1,1,0)

    def ARM(self,c1,c2,c3):
        _,_,s3,s4=c2.size()
        c2=self.ARM_layer1(F.avg_pool2d(c2,kernel_size=(s3,s4)))*c2+F.upsample_bilinear(c1,scale_factor=2)
        _, _, s3, s4 = c3.size()
        c3 = self.ARM_layer2(F.avg_pool2d(c3,kernel_size=(s3,s4))) * c3 + F.upsample_bilinear(c2,scale_factor=2)
        return c3

    def FFM(self,c_feat,s_out):
        feat=torch.cat([c_feat,s_out],1)
        feat=self.FFM_layer1(feat)
        _,_,s3,s4=feat.size()
        feat = self.FFM_layer2(F.avg_pool2d(feat,kernel_size=(s3,s4))) * feat+feat
        return feat

    def forward(self, x):
        context_input=x

        c1,c2,c3=self.Context_model(context_input)

        c1=self.enc1(c1)
        c2=self.enc2(c2)
        c_feat=self.ARM(c1,c2,c3)
        x=self.Spatial_entry(x)

        for layer in self.Spatial_model:
            x=layer(x)

        s_out=self.enc3(x)

        feat=self.FFM(c_feat,s_out)

        score=self.classifer(feat)

        score=F.upsample_bilinear(score,scale_factor=8)

        return score
