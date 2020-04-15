'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch, os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchsummary import summary

import torchvision
import torchvision.transforms as transforms

import torchvision.datasets as datasets
import torch.utils.data as data

import torchvision.models as models


import os
import argparse

from models import *
from utils import progress_bar
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', help='resume from checkpoint',
                    default='checkpoint/resnet101_1_1_ckpt.t8')
args = parser.parse_args()

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# # Data
# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(224, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#Tiny Imagenet dataset
data_dir = './data/vsmall-imagenet/'
num_workers = {'train' : 1,'val'   : 1,'test'  : 1}
data_transforms = {
    'train': transforms.Compose([
     transforms.RandomCrop(224, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
 ]),
    'val': transforms.Compose([
     transforms.RandomCrop(224, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
 ]),
    'test': transforms.Compose([
     transforms.RandomCrop(224, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
 ])
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val','test']}

a = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
print(a.classes)
exit(1)
#print(image_datasets)


#dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=num_workers[x]) for x in ['train', 'val', 'test']}
dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=num_workers[x]) for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

print(image_datasets['test'])

exit(1)
trainloader = dataloaders['train']
testloader = dataloaders['test']

print(trainloader, testloader)

print(len(trainloader.dataset))

dataiter = iter(trainloader)

#print(dataiter.next())
#inputs, classes = next(iter(dataloaders['train']))

exit(1)


# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = ResNet101()

net = torchvision.models.vgg19(pretrained=True)#models.resnet18()#torchvision.models.ResNet18(pretrained=True)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Loading pretrained checkpoint..')
#     #print('==> Resuming from checkpoint..')
#     #assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     #checkpoint = torch.load(args.resume)
#     checkpoint =
#
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']


summary(net, (3,224,224))
exit(1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)



# for name in net.parameters():
#     print(name)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

    
def extract_features(mode):
    #FIXME: what is top-k?
    table = np.load('results/topk.npy')
    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


    #print(trainloader)
    #exit(1)
    loader = {'train': trainloader, 'test': testloader}[mode]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            print(inputs)
            exit(1)
            outputs = net.module.forward_cam(inputs)[4]

            list_outputs = []

            for i, output in enumerate(outputs):
                features = output[table[targets[i]]]

                list_outputs.append(features)

            features = torch.stack(list_outputs)


            #FIXME: check output with pdb.trace
            max_features = F.adaptive_max_pool2d(features, 1)
            features = features / (max_features + 1e-16) # this is the attention maps

            features_ = F.interpolate(features, inputs.shape[-2:], mode='bilinear', align_corners=False)

            mask = torch.where(features_ > 0.5, torch.ones_like(features_), torch.zeros_like(features_))

            masked_inputs = inputs.unsqueeze(1) * mask.unsqueeze(2)

            list_masked_outputs = []

            for i, masked_input in enumerate(masked_inputs): 
                masked_output = net(masked_input).squeeze()
                
                list_masked_outputs.append(masked_output)

            masked_outputs = torch.stack(list_masked_outputs) # this is the feature output w.r.t the highest frequency features of that class

    

if __name__ == '__main__':
    #extract_features('train')
    #extract_features('val')
    extract_features('test')

    # for epoch in range(start_epoch, start_epoch+200):
    #     train(epoch)
    #     test(epoch)
