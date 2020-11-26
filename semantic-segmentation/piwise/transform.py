import torch
import numpy as np
from PIL import Image

colors = [    
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

def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    for i in np.arange(n):
        r, g, b = np.zeros(3)
        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))
        cmap[i,:] = np.array([r, g, b])
    return cmap

class Relabel:
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class ToLabel:
    def __call__(self, image):
        a=torch.from_numpy(np.array(image)).long().unsqueeze(0)
        return a

class Colorize:
    def __init__(self, n=22):
        self.cmap = colors
        self.cmap = torch.from_numpy(np.array(self.cmap)).byte()    
    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3,size[0], size[1]).fill_(0)
        for label in range(0, 19):
            mask=gray_image==label
            color_image[0,mask] = self.cmap[label][0]
            color_image[1,mask] = self.cmap[label][1]
            color_image[2,mask] = self.cmap[label][2]
        return color_image
     
