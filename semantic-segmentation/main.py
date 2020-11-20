import os
import torch
from argparse import ArgumentParser
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage,Resize,RandomRotation,RandomGrayscale
from piwise.dataset import Cityscapes
from piwise.dataset import test_set

from piwise.network import FPN_ASPP,Nonlocal_Network,BiseNetv1
from piwise.criterion import CrossEntropyLoss2d
from piwise.transform import  Colorize

import torchvision.transforms as transforms
from PIL import Image
import time

torch.backends.cudnn.benchmark = True 

to_tensor=transforms.ToTensor()
color_transform = Colorize()
image_transform = ToPILImage()
to_img=transforms.ToPILImage()

input_transform = Compose([ 
    Resize((512,1024)),
    RandomGrayscale(0.02),
    CenterCrop((512,1024)),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

target_transform = Compose([   
    Resize((512,1024)),
    CenterCrop((512,1024)),
    ])

def train(args, model):

    loader = DataLoader(Cityscapes(args.datadir, input_transform, target_transform, scale=0),\
                        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    model.train()
    weight = torch.ones(20)
    weight[19]=0
    criterion = CrossEntropyLoss2d(weight.cuda())
    model.load_state_dict(torch.load('model-001-0000.pth'),strict=True)
    lr_start = 0.05
    for epoch in range(0, args.num_epochs):
        lr_init=lr_start*((args.num_epochs-epoch)/args.num_epochs)
        optimizer=SGD(model.parameters(),lr=lr_init,momentum=0.9,weight_decay=0.0005)
        epoch_loss = []
        for step, (images,label) in enumerate(loader):
            inputs = Variable(images).cuda()
            targets = Variable(label).cuda()
            output= model(inputs)
            def loss_back(loss,optimizer,args,step):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()     
                epoch_loss.append(loss.data)
                average1 = sum(epoch_loss) / len(epoch_loss)
                print(f'step:{step},loss:{average1}')
                if (epoch+1)%1 ==0:
                    if args.steps_save > 0 and step % args.steps_save == 0:
                       filename = f'{args.model}-{epoch:03}-{step:04}.pth'
                       torch.save(model.state_dict(), filename)
            loss = criterion(output,targets)
            loss_back(loss,optimizer,args,step)

from piwise.utils import trans_id
def evaluate( model):
    # model.load_state_dict(torch.load('segnet-024-0000.pth'),strict=False)
    loader = DataLoader(test_set('D:\someprogram\dataset\cityscapes_test\\', input_transform, target_transform),
                        num_workers=0, batch_size=1, shuffle=False)

    save_dir1 = './gray_results/'
    save_dir2 = './rgb_labels/'
    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)
    model.eval().cuda()
    for param in model.parameters():
        param.requires_grad = False
    T=0
    for i ,(image,path) in enumerate(loader):

        image = Variable(image.cuda(), volatile=True)
        s1=time.time()
        outputs=model(image)
        s2=time.time()
        T+=s2-s1

        outputs=outputs[0]
        out=outputs.data.max(1)[1][0]
        rgb=image_transform(color_transform(out))
        out=out.cpu().numpy()
        out=trans_id(out)
        gray=Image.fromarray(out.astype('uint8'))
        gray.save(save_dir1+path[0])
        rgb.save(save_dir2+path[0])

def main(args):
    Net = None
    if args.model == 'bisenetv1':
        Net = BiseNetv1
    if args.model == 'Nonlocal_Network':
        Net = Nonlocal_Network
    if args.model == 'FPN_ASPP':
        Net = FPN_ASPP
    assert Net is not None, f'model {args.model} not available'

    model = Net(args.num_classes)

    if args.cuda:
        model = model.cuda()
    if args.mode == 'eval':
        evaluate(model)
    if args.mode == 'train':
        train(args, model)
 
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', required=True)
    parser.add_argument('--state')
    parser.add_argument('--num-classes',type=int,default=20)
    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True
    parser_eval = subparsers.add_parser('eval')
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--port', type=int, default=80)
    parser_train.add_argument('--datadir', required=True)
    parser_train.add_argument('--num-epochs', type=int, default=32)
    parser_train.add_argument('--num-workers', type=int, default=4)
    parser_train.add_argument('--batch-size', type=int, default=1)
    main(parser.parse_args())

