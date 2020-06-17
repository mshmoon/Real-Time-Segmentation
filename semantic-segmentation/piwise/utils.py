import numpy as np
from PIL import Image
from torchvision.transforms import Compose, CenterCrop
from torchvision.transforms import ToTensor, ToPILImage,Resize,RandomRotation,RandomGrayscale

def trans_id(pred_label):
    ignore_label=250
    id_to_transform = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
    mask = np.zeros((1024//2, 2048//2))
    for k, v in id_to_transform.items():
        mask[pred_label == v] = k
    return mask

def to_crop(x, root):
    image = Image.open('data/images/' + root.split('images')[1] + '.jpg')
    tensor = ToTensor()(image)
    v1, v2, v3 = tensor.size()
    to_crop =CenterCrop((v2, v3))
    x = to_crop(x)
    x.save('data/output_labels/' + root.split('images')[1] + '.jpg')

def to_trans(x, path):
    out = color_transform(x[0].data.max(0)[1])
    image = ToPILImage()(out)
    image.show()
    root = (path[0])
    image.save('results/' + path[0])
