#These are the python libraries that will be used
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.models as models
from PIL import Image
import random
from deepfool import deepfool
from TargetX import deepfool_hybrid2, deepfool_hybrid2_arg
import os
import time
import glob
import cv2
import csv

#these are the needed networks that are to be used
res34 = models.resnet34(pretrained=True)
res101 = models.resnet101(pretrained=True)
gnet = models.googlenet(pretrained=True)
anet = models.alexnet(pretrained=True)
vgg = models.vgg19(pretrained=True)
d201 = models.densenet201(pretrained=True)


def TestingFunction(net, eps, targetcsv):
    net.eval()
    net.cuda()

    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]

    fieldnames = ['Image', 'Original Label', 'Classified Label Before Perturbation', 'Targeted Label', 'Perturbed Label', 'Execution Time', 'F_k', 'Avg Difference', 'Frobenius of Difference']

    with open(targetcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fieldnames)

    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')
    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    Accuracy = 0
    PerturbedAccuracy = 0
    TargetSuccess = 0
    PerturbedAvgFk = 0
    PerturbedAvgDiff = 0
    PerturbedAvgFroDiff = 0
    counter = 0

    for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'): #assuming jpg
        im_orig = Image.open(filename).convert('RGB')
        print(filename[47:75])
        if counter == 1000:
            break

        im = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)])(im_orig)

        inputNum = random.randint(0, 999)
        print('Generated Eval Label: ', inputNum)
        correct = ILSVRClabels[np.int(counter)].split(' ')[1]
        x = Variable(im.cuda()[None, :], requires_grad=True)
        detect = net.forward(x)
        detected = np.argmax(detect.data.cpu().numpy().flatten())
        str_label_detect = labels[np.int(detected)].split(',')[0]
        print('Detected: ', str_label_detect)
        while(inputNum == int(correct) or inputNum == detected):
            inputNum = random.randint(0, 999)
            print('Eval same as correct/detected. Generating new input: ', inputNum)
        str_label_correct = labels[np.int(correct)].split(',')[0]
        str_label_target = labels[np.int(inputNum)].split(',')[0]
        print('Target: ', str_label_target)
        print('Correct: ', str_label_correct)

        start_time = time.time()
        r, loop_i, label_orig, label_pert, pert_image, newf_k = deepfool_hybrid2_arg(im, net, inputNum, eps)
        end_time = time.time()
        execution_time = end_time - start_time
        print("execution time = " + str(execution_time))
        str_label_orig = labels[np.int(label_orig)].split(',')[0]
        str_label_pert = labels[np.int(label_pert)].split(',')[0]
        print("Original label: ", str_label_orig)
        print("Perturbed label: ", str_label_pert)
        if (int(label_orig) == int(correct)):
            print("Classifier is correct")
            Accuracy = Accuracy + 1
        if (int(label_pert) == int(correct)):
            print("Classifier is correct on perturbed image")
            PerturbedAccuracy = PerturbedAccuracy + 1
        if (int(label_pert) == inputNum):
            print("Image perturbed to target")
            TargetSuccess = TargetSuccess + 1

        def clip_tensor(A, minv, maxv):
            A = torch.max(A, minv * torch.ones(A.shape))
            A = torch.min(A, maxv * torch.ones(A.shape))
            return A


        clip = lambda x: clip_tensor(x, 0, 255)

        tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                                 transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                 transforms.Lambda(clip),
                                 transforms.ToPILImage(),
                                 transforms.CenterCrop(224)])

        imagetransform = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                                             transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                             transforms.Lambda(clip)])

        tensortransform = transforms.Compose([transforms.Scale(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                                              transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                              transforms.Lambda(clip)])
        diff = imagetransform(pert_image.cpu()[0]) - tensortransform(im_orig)
        # Calculate frobenius of difference
        fro = np.linalg.norm(diff.numpy())
        # Calculate average distance
        average = torch.mean(torch.abs(diff))
        csvrows = []
        PerturbedAvgFk = PerturbedAvgFk + newf_k
        PerturbedAvgDiff = PerturbedAvgDiff + average
        PerturbedAvgFroDiff = PerturbedAvgFroDiff + fro
        #Append values to rows, append to csv file
        csvrows.append([filename[47:75], str_label_correct, str_label_orig, str_label_target, str_label_pert, torch.cuda.memory_stats('cuda:0')['active.all.current'], str(loop_i), str(execution_time), newf_k, average, fro])
        with open(targetcsv, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(csvrows)
        print('-------------------------------------------')
        counter = counter + 1

    with open(targetcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 1000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(PerturbedAccuracy/1000)])
        csvwriter.writerows(["Perturbed Target Success: " + str(TargetSuccess / 1000)])
        csvwriter.writerows(["Avg F_k: " + str(PerturbedAvgFk/1000)])
        csvwriter.writerows(["Avg Difference: " + str(PerturbedAvgDiff / 1000)])
        csvwriter.writerows(["Avg Frobenius of Difference: " + str(PerturbedAvgFroDiff / 1000)])


#TestingFunction(res34, 0.05, 'targetedapproachresnet34eps005.csv')
#TestingFunction(res101, 0.05, 'targetedapproachresnet101eps005.csv')
#TestingFunction(gnet, 0.05, 'targetedapproachgoogleneteps005.csv')
#TestingFunction(anet, 0.05, 'targetedapproachalexneteps005.csv')
#TestingFunction(vgg, 0.05, 'targetedapproachvggnet19eps005.csv')
#TestingFunction(d201, 0.05, 'targetedapproachdensenet201eps005.csv')