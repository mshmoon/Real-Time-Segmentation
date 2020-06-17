import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

EXTENSIONS = ['.jpg', '.png']

num_classes = 19
ignore_label = 19


def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class Cityscapes(Dataset):
    def __init__(self, root, input_transform=None, target_transform=None,scale=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')
        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.i=scale
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
       
        filename = self.filenames[index]
        filename1=filename.split("gtFine")[0]+'leftImg8bit'
        degree = int(np.random.uniform(0,360,1))

        ind=int(np.random.uniform(0,5,1))
        scale=[1/2,3/4,1,5/4,3/2]
                  
        with open(image_path(self.images_root, filename1, '.png'), 'rb') as f:
            image = load_image(f).convert('RGB')
            s1,s2,s3=np.shape(image)                   
            #image=image.resize((int(s1*scale[ind]),int(s2*scale[ind])))
            if degree>180:
                image=image.transpose(Image.FLIP_LEFT_RIGHT)

        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            
            label = load_image(f).convert('P')  
            #label=label.resize((int(s1*scale[ind]),int(s2*scale[ind])))
            if degree>180:
                label=label.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.array(label)
            mask_copy = mask.copy()
            for k, v in self.id_to_trainid.items():
                mask_copy[mask == k] = v
            label = Image.fromarray(mask_copy.astype(np.uint8))
        if self.input_transform is not None:
            image = self.input_transform(image)    
        if self.target_transform is not None:
            label = self.target_transform(label)
        label=torch.from_numpy(np.array(label)).long()
        return image, label
    def __len__(self):
        return len(self.filenames)
        

class test_set(Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.filenames = [image_basename(f)
            for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames.sort()
        self.input_transform = input_transform
    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        if self.input_transform is not None:
            image = self.input_transform(image)
        save_path = filename+".png"
        return image, save_path
    def __len__(self):
        return len(self.filenames)


