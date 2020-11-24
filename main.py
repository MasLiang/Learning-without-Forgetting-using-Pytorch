'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import scipy.io as scio
from scipy.io import loadmat
import torchvision
import os
import argparse
from utils import progress_bar
import numpy as np
import random

import matplotlib.pyplot as plt
import pdb
import transforms
from dataset import CUB_200_2011_Train, CUB_200_2011_Test
import torchvision.transforms as tfs
import torchvision.datasets as datasets

def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
parser = argparse.ArgumentParser(description='Learning Without Forgetting')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch

# New Data
print('==> Preparing data..')
train_transforms = transforms.Compose([
        transforms.ToCVImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.48560741861744905, 0.49941626449353244, 0.43237713785804116],
            [0.2321024260764962, 0.22770540015765814, 0.2665100547329813])
    ])

test_transforms = transforms.Compose([
        transforms.ToCVImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.4862169586881995, 0.4998156522834164, 0.4311430419332438],
            [0.23264268069040475, 0.22781080253662814, 0.26667253517177186])
    ])

trainset = CUB_200_2011_Train(
        '~/CUB_200_2011', 
        transform=train_transforms,
    )
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=384, shuffle=True, num_workers=2)

testset = CUB_200_2011_Test(
        '~/CUB_200_2011', 
        transform=test_transforms,
    )
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Used to test the performance on old dataset
normalize = tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
valloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root='~/Imagenet2012/ILSVRC2012_img_val', transform=tfs.Compose([
            tfs.Resize(256),
            tfs.CenterCrop(224),
            tfs.ToTensor(),
            normalize,
            ])),
        batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True)
# Model
print('==> Building model..')
net = torchvision.models.alexnet(pretrained=False)
net11 = torchvision.models.alexnet(pretrained=False)

# Use the pretrained model from Pytorch
oor = torch.load('~/pytorch_pretrained_modle/alexnet_pretrained.pth')
# 'Net'  is the new model to learn new classes
net.load_state_dict(oor)
# 'Net11' is the old model to learn old classes
net11.load_state_dict(oor)
# Number of new class, if this number is changed, you should change 'dataset.py'
# You can chech the 31th line and the 90th line of dataset.py
# In my experiment, I test each 200 classes on by on and retain this value as '1'
# If you want to add more classes, you should change the code I said before
num_new_class = 1
# Old number of input/output channel of the last FC layer in old model
in_features = net.classifier[6].in_features
out_features = net.classifier[6].out_features
# Old weight/bias of the last FC layer
weight = net.classifier[6].weight.data
bias = net.classifier[6].bias.data
# New number of output channel of the last FC layer in new model
new_out_features = num_new_class+out_features
# Creat a new FC layer and initial it's weight/bias
new_fc = nn.Linear(in_features, new_out_features)
kaiming_normal_init(new_fc.weight)
new_fc.weight.data[:out_features] = weight
new_fc.bias.data[:out_features] = bias
# Replace the old FC layer
net.classifier[6] = new_fc
# CUDA
net = net.to(device)
net11 = net11.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    net11 = torch.nn.DataParallel(net11)
    cudnn.benchmark = True

# Loss function
criterion = nn.CrossEntropyLoss()
# Temperature of the new softmax proposed in 'Distillation of Knowledge'
T=2
# Used to balance the new class loss1 and the old class loss2
# Loss1 is the cross entropy between output of the new task and label
# Loss2 is the cross entropy between output of the old task and output of the old model
# It should be noticed that before calculating loss2, the output of each model should- 
# -be handled by the new softmax 
alpha = 0.01
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.eval()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        targets += out_features
        optimizer.zero_grad()
        outputs = net(inputs)
        soft_target = net11(inputs)
        # Cross entropy between output of the new task and label
        loss1 = criterion(outputs,targets)
        # Using the new softmax to handle outputs
        outputs_S = F.softmax(outputs[:,:out_features]/T,dim=1)
        outputs_T = F.softmax(soft_target[:,:out_features]/T,dim=1)
        # Cross entropy between output of the old task and output of the old model
        loss2 = outputs_T.mul(-1*torch.log(outputs_S))
        loss2 = loss2.sum(1)
        loss2 = loss2.mean()*T*T
        loss = loss1*alpha+loss2*(1-alpha)
        loss.backward(retain_graph=True)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/(batch_idx+1)
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets += out_features
            outputs = net(inputs)
            soft_target = net11(inputs)
            loss1 = criterion(outputs,targets)
            loss = loss1
            outputs_S = F.softmax(outputs[:,:out_features]/T,dim=1)
            outputs_T = F.softmax(soft_target[:,:out_features]/T,dim=1)
            loss2 = outputs_T.mul(-1*torch.log(outputs_S))
            loss2 = loss2.sum(1)
            loss2 = loss2.mean()*T*T
            loss = loss1*alpha+loss2*(1-alpha)

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
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return acc
def val(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted_old = outputs.max(1)
            total += targets.size(0)
            correct += predicted_old.eq(targets).sum().item()
            progress_bar(batch_idx, len(valloader), 'Acc: %.3f%% (%d/%d)'
                         % (100.*correct/total,  correct, total))
    return 100.*correct/total

epochs = []
test_new_accs = []
test_old_accs = []
train_losses = []
layer_num = [0,3,7,10,14,17,20,24,27,30,34,37,40]
ct = 0


# Ensure that the old model don't train
for param in net11.module.parameters():
    param.requires_grad = False

## Warm up step
## In my experiment, I didn't use this step, and the paper said that this step is not necessary
#
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
#         momentum=0.9, weight_decay=5e-4)
#
## Frozen the model
# for param in net11.module.parameters():
#     param.requires_grad = False
## Thaw the last FC layer
# for param in net.module.classifier[6].parameters():
#     param.requires_grad = True
#
# for epoch in range(start_epoch, start_epoch+20):
#     train_loss = train(epoch)
#     # Make sure there are only weights/bias corresponding to the new task being trained  
#     net.module.classifier[6].weight.data[:out_features] = net11.module.classifier[6].weight.data
#     net.module.classifier[6].bias.data[:out_features] = net11.module.classifier[6].bias.data
#     acc,test_loss = test(epoch)
## Thaw the model
# for param in net.module.parameters():
#     param.requires_grad = True

## train step
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
        momentum=0.9, weight_decay=5e-4)

for epoch in range(start_epoch, start_epoch+200):
    train_loss = train(epoch)
    acc_new = test(epoch)
    acc_old = val(epoch)
    test_new_accs.append(acc_new)
    test_old_accs.append(acc_old)
    train_losses.append(train_loss)
    epochs.append(epoch)
## Save the final model
# torch.save(net.state_dict(), 'temp.pkl')

# Picture of the new class test accuracy changing with training
plt.figure()
plt.plot(epochs,test_new_accs)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title('new class test accuracy')
plt.savefig('./acc_new_class.jpg')

# Picture of the old class test accuracy changing with training
plt.figure()
plt.plot(epochs,test_old_accs)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title('old class test accuracy')
plt.savefig('./acc_old_class.jpg')

# Picture of the training loss changing with training
plt.figure()
plt.plot(epochs,train_losses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('train loss')       
plt.savefig('./train_loss.jpg')

