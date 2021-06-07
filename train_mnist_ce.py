from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import time
from model import*
import sys


if_train = False
checkpoint = './ckpt/mnist_ce.pth.tar'

train_dataset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


def lr_scheduler(optimizer, epoch, lr_decay_epoch=35):
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


snn = MNIST().cuda()
optimizer = torch.optim.SGD(snn.parameters(), lr=1e-2, momentum=0.9)
criterion = nn.CrossEntropyLoss().cuda()  

num_epochs = 50
best_acc = 0  

if not if_train:
    num_epochs = 1
    snn.load_state_dict(torch.load(checkpoint))


for epoch in range(num_epochs):

    if if_train:
        snn.train()
        running_loss = 0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            snn.zero_grad()
            optimizer.zero_grad()

            images = images.float().cuda()
            labels = labels.cuda()

            outputs = snn(images)
            loss = criterion(outputs, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()

            if (i+1) % 300 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f' % (
                epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, running_loss / 300))
                running_loss = 0
                print('Time elasped:', time.time()-start_time)

    snn.eval()
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, 35)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = snn(inputs)
            loss = criterion(outputs, labels)

            _, predicted = outputs.cpu().max(1)
            total += float(labels.size(0))

            correct += float(predicted.eq(labels.cpu()).sum().item())
            if batch_idx %100 ==0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

    acc = 100. * float(correct) / float(total)
    print('Iters:', epoch)
    print('Test Accuracy of the model on the test images: %.3f' % acc)

    if if_train:
        if acc > best_acc:
            best_acc = acc
            print('Saving\n')
            torch.save(snn.state_dict(), checkpoint)
