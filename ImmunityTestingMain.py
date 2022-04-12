import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.optim as optim
from ImmunityTestingFunction import TargetX_Immunity_Testing, targetFGSM_Immunity_Testing
import torchvision.models as models
from PIL import Image
from TargetX import targetx_arg, targetx_return_I_array
import os
import glob
import random
import art

#Check if cuda is available.
is_cuda = torch.cuda.is_available()
device = 'cpu'

#If cuda is available use GPU for faster processing, if not, use CPU.
if is_cuda:
    print("Using GPU")
    device = 'cuda:0'
else:
    print("Using CPU")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

global new_fk

# Define the network to be finetuned and use to train
anet = models.alexnet(pretrained=True)
gnet = models.googlenet(pretrained=True)
rnet_34 = models.resnet34(pretrained=True)
rnet_101 = models.resnet101(pretrained=True)

# Put network on GPU
if is_cuda:
    anet.cuda()
    gnet.cuda()
    rnet_34.cuda()
    rnet_101.cuda()

# Set network to evaluation mode
anet.eval()
gnet.eval()
rnet_34.eval()
rnet_101.eval()

# for testing different models
# net = anet
# net = gnet
net = rnet_34
# net = rnet_101

ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

# Define networks to be finetuned for each approach, load them into the GPU, and set them all in training mode

# TARGET-X AlexNet
targetX_anet_0005 = models.alexnet(pretrained=True)
targetX_anet_0005.cuda()
targetX_anet_0005.eval()

targetX_anet_05 = models.alexnet(pretrained=True)
targetX_anet_05.cuda()
targetX_anet_05.eval()

targetX_anet_2 = models.alexnet(pretrained=True)
targetX_anet_2.cuda()
targetX_anet_2.eval()

# TARGET-X GoogleNet
targetX_gnet_0005 = models.googlenet(pretrained=True)
targetX_gnet_0005.cuda()
targetX_gnet_0005.eval()

targetX_gnet_05 = models.googlenet(pretrained=True)
targetX_gnet_05.cuda()
targetX_gnet_05.eval()

targetX_gnet_2 = models.googlenet(pretrained=True)
targetX_gnet_2.cuda()
targetX_gnet_2.eval()

# TARGET-X ResNet-34
targetX_rnet_34_0005 = models.resnet34(pretrained=True)
targetX_rnet_34_0005.cuda()
targetX_rnet_34_0005.eval()

targetX_rnet_34_05 = models.resnet34(pretrained=True)
targetX_rnet_34_05.cuda()
targetX_rnet_34_05.eval()

targetX_rnet_34_2 = models.resnet34(pretrained=True)
targetX_rnet_34_2.cuda()
targetX_rnet_34_2.eval()

# TARGET-X ResNet-101
targetX_rnet_101_0005 = models.resnet101(pretrained=True)
targetX_rnet_101_0005.cuda()
targetX_rnet_101_0005.eval()

targetX_rnet_101_05 = models.resnet101(pretrained=True)
targetX_rnet_101_05.cuda()
targetX_rnet_101_05.eval()

targetX_rnet_101_2 = models.resnet101(pretrained=True)
targetX_rnet_101_2.cuda()
targetX_rnet_101_2.eval()

# TARGET-FGSM AlexNet
targetfgsm_anet_0005 = models.alexnet(pretrained=True)
targetfgsm_anet_0005.cuda()
targetfgsm_anet_0005.eval()

targetfgsm_anet_05 = models.alexnet(pretrained=True)
targetfgsm_anet_05.cuda()
targetfgsm_anet_05.eval()

targetfgsm_anet_2 = models.alexnet(pretrained=True)
targetfgsm_anet_2.cuda()
targetfgsm_anet_2.eval()

# TARGET-FGSM GoogleNet
targetfgsm_gnet_0005 = models.googlenet(pretrained=True)
targetfgsm_gnet_0005.cuda()
targetfgsm_gnet_0005.eval()

targetfgsm_gnet_05 = models.googlenet(pretrained=True)
targetfgsm_gnet_05.cuda()
targetfgsm_gnet_05.eval()

targetfgsm_gnet_2 = models.googlenet(pretrained=True)
targetfgsm_gnet_2.cuda()
targetfgsm_gnet_2.eval()

# TARGET-FGSM ResNet-34
targetfgsm_rnet_34_0005 = models.resnet34(pretrained=True)
targetfgsm_rnet_34_0005.cuda()
targetfgsm_rnet_34_0005.eval()

targetfgsm_rnet_34_05 = models.resnet34(pretrained=True)
targetfgsm_rnet_34_05.cuda()
targetfgsm_rnet_34_05.eval()

targetfgsm_rnet_34_2 = models.resnet34(pretrained=True)
targetfgsm_rnet_34_2.cuda()
targetfgsm_rnet_34_2.eval()

# TARGET-FGSM ResNet-101
targetfgsm_rnet_101_0005 = models.resnet101(pretrained=True)
targetfgsm_rnet_101_0005.cuda()
targetfgsm_rnet_101_0005.eval()

targetfgsm_rnet_101_05 = models.resnet101(pretrained=True)
targetfgsm_rnet_101_05.cuda()
targetfgsm_rnet_101_05.eval()

targetfgsm_rnet_101_2 = models.resnet101(pretrained=True)
targetfgsm_rnet_101_2.cuda()
targetfgsm_rnet_101_2.eval()


def TargetX_Immunity_Training(orig_net, immune_net, name, eps):
    if is_cuda:
        immune_net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(immune_net.parameters(), lr=1e-5)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for epoch in range(5):
        if epoch != 0:
            PATH = './targetX_adv_net' + name + str(epoch) + '.pth'
            torch.save(immune_net.state_dict(), PATH)
            immune_net.eval()
            csv = 'targetX' + name + 'immunityepoch' + str(epoch) + str(eps) + '.csv'
            TargetX_Immunity_Testing(net, immune_net, eps, csv)
            immune_net.train()
        running_loss = 0.0
        i = 0
        counter = 0
        for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):
            if counter == 5000:
                break
            im_orig = Image.open(filename).convert('RGB')
            im = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])(im_orig)
            # generate a random label id for the targeted algorithm from [0-999]
            n = random.randint(0, 999)
            # prints label to validate with alg
            print(n)
            # returns the 10 nearest labels for the image
            I = targetx_return_I_array(im, orig_net, 10)
            print(I)

            # label to exclude from choice
            exclude_label = I[0]
            # chooses a random int in the I array
            inputNum = random.choice(I)
            if inputNum == exclude_label:
                print("target label same as original label")
                inputNum = random.choice(I)
            r, loop_i, label_orig, label_pert, pert_image, newf_k = targetx_arg(im, orig_net, inputNum, eps)
            inputs = pert_image
            labels = ILSVRClabels[np.int(counter)].split(' ')[1]
            labels = torch.tensor([int(labels)])
            labels = labels.to('cuda')
            # zero the param gradients
            optimizer.zero_grad()
            # forward and backward propagation and then optimization
            outputs = immune_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # printing the statistics
            running_loss += loss.item()
            # prints every 2000 mini-batches
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            i = i + 1
            counter = counter + 1
    print('Finished Training')
    PATH = 'targetX_adv_net' + str(eps) + name + '.csv'
    torch.save(immune_net.state_dict(), PATH)
    return immune_net

def TargetFGSM_Immunity_Training(orig_net, immune_net, name, eps):
    if is_cuda:
        immune_net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(immune_net.parameters(), lr=1e-5)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(orig_net.parameters(), lr=0.01)

    classifier = art.estimators.classification.PyTorchClassifier(
        model=orig_net,
        input_shape=(3, 224, 224),
        loss=criterion,
        optimizer=optimizer,
        nb_classes=1000
    )

    for epoch in range(5):
        if epoch != 0 :
            PATH = './targetFGSM_adv_net' + name + str(epoch) + '.pth'
            torch.save(immune_net.state_dict(), PATH)
            immune_net.eval()
            csv = 'targetFGSM' + name + 'immunityepoch' + str(epoch) + str(eps) + '.csv'
            targetFGSM_Immunity_Testing(net, immune_net, eps, csv)
            immune_net.train()
        running_loss = 0.0
        i = 0
        counter = 0
        for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):
            if counter == 5000:
                break
            im_orig = Image.open(filename).convert('RGB')
            im = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])(im_orig)

            I = targetx_return_I_array(im, orig_net, 10)
            print(I)

            # label to exclude from choice
            exclude_label = I[0]
            # chooses a random int in the I array
            inputNum = random.choice(I)
            if inputNum == exclude_label:
                print("target label same as original label")
                inputNum = random.choice(I)
            targetLabel = np.array([])
            targetLabel = np.append(targetLabel, inputNum)

            input_batch = im.unsqueeze(0)
            result = classifier.predict(input_batch, 1, False)
            label_orig = np.argmax(result.flatten())

            labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

            attack = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=eps, norm=np.inf, targeted=True)
            input_array = input_batch.numpy()
            img_adv = attack.generate(x=input_array, y=targetLabel)

            result_adv = classifier.predict(img_adv, 1, False)
            label_pert = np.argmax(result_adv.flatten())

            inputs = result_adv
            labels = ILSVRClabels[np.int(counter)].split(' ')[1]
            labels = torch.tensor([int(labels)])
            labels = labels.to('cuda')
            # zero the param gradients
            optimizer.zero_grad()
            # forward and backward propagation and then optimization
            outputs = immune_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # printing the statistics
            running_loss += loss.item()
            # prints every 2000 mini-batches
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            i = i + 1
            counter = counter + 1
    print("Finished Training")
    PATH = 'targetFGSM_adv_net' + str(eps) + name + '.csv'
    torch.save(immune_net.state_dict(), PATH)
    return immune_net
