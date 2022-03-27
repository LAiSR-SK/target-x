from ast import operator
from itertools import count
from unittest import result
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from TargetX import targetx_arg, targetx_return_I_array
import os
import time
import glob
from imagenet_labels import classes
import cv2
import csv
import random

#Check if cuda is available.
is_cuda = torch.cuda.is_available()
device = 'cpu'

#If cuda is available use GPU for faster processing, if not, use CPU.
if is_cuda:
    print("Using GPU")
    device = 'cuda:0'
else:
    print("Using CPU")

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

global new_fk

def TargetX_Immunity_Testing(orig_net, targetX_net, eps, csvfilename):

    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    accuracy = 0
    targetx_accuracy = 0
    targetx_immunity = 0
    targetx_targeted_immunity = 0
    targetx_targeted_result = 0
    targetx_csv = csvfilename
    fieldnames = ['Image', 'Correct Label', 'Classified Label Before Perturbation', 'Perturbed Label', 'Label from Immune Network']

    with open(targetx_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(fieldnames)

    ILSVRC_labels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

    counter = 4999
    for filename in glob.glob('D:/imageNet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):
        if int(filename[82:86]) < 5000:
            continue
        im_orig = Image.open(filename).convert('RGB')
        if counter == 10000:
            break
        im = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])(im_orig)

        start_time = time.time()

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

        print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution Time = " + str(execution_time))

        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
        correct = ILSVRC_labels[np.int(counter)].split(' ')[1]

        str_label_orig_correct = labels[np.int(correct)].split(',')[0]
        str_label_orig_orig = labels[np.int(label_orig)].split(',')[0]
        str_label_orig_pert = labels[np.int(label_pert)].split(',')[0]

        print("Correct label = ", str_label_orig_correct)
        print("Original label (Original Network) = ", str_label_orig_orig)
        print("Perturbed label (Original Network) = ", str_label_orig_pert)

        result_set = targetX_net(im[None, :].cuda())
        result = np.argmax(result_set.detach().cpu().numpy())
        str_label_result = labels[np.int(result)].split(',')[0]
        print("Result from Immune Network = ", str_label_result)

        if (int(label_orig) == int(correct)):
            print("Original Classifier is Correct")
            accuracy = accuracy + 1

        if (int(result) == int(correct)):
            print("Immune Classifier is Correct")
            targetx_accuracy = targetx_accuracy + 1

        if (int(result) == int(label_orig)):
            print("Immune Classifier is the same as the Original Classification")
            targetx_immunity = targetx_immunity + 1

        if (int(label_pert) == int(inputNum)):
            print("Image correctly perturbed to target label on Original Net.")
            targetx_targeted_result = targetx_targeted_result + 1

        if (int(result) == int(inputNum)):
            print("Image correctly perturbed to target label on Immune Net.")
            targetx_targeted_immunity = targetx_targeted_immunity + 1

        targetx_rows = []
        targetx_rows.append([filename[47:75], str_label_orig_correct, str_label_orig_orig, str_label_orig_pert, str_label_result])

        with open(targetx_csv, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(targetx_rows)
        counter = counter + 1

    with open(targetx_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(["Epsilon: " + str(eps)])
        csv_writer.writerows(["Original Accuracy: " + str(accuracy / 5000)])
        csv_writer.writerows(["Perturbed Accuracy: " + str(targetx_accuracy / 5000)])
        csv_writer.writerows(["Untargeted Robustness: " + str(targetx_immunity / 5000)])
        csv_writer.writerows(["Perturbation Success: " + str(targetx_targeted_result / 5000)])
        csv_writer.writerows(["Targeted Robustness: " + str(targetx_targeted_immunity / 5000)])
