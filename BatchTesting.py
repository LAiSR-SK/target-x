# These are the python libraries that will be used
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from PIL import Image
import random
from TargetX import targetx_arg, targetx_return_I_array
import os
from imagenet_labels import classes
import time
import glob
import csv
import art
import cv2

# these are the needed networks that are to be used
res34 = models.resnet34(pretrained=True)
res101 = models.resnet101(pretrained=True)
gnet = models.googlenet(pretrained=True)
anet = models.alexnet(pretrained=True)
vgg = models.vgg19(pretrained=True)
d201 = models.densenet201(pretrained=True)


def runBatchTargetX(net, eps, targetcsv):
    net.eval()
    net.cuda()

    # Set mean and standard deviation for normalizing image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define fieldnames for CSV
    fieldnames = ['Image', 'Original Label', 'Classified Label Before Perturbation', 'Targeted Label',
                  'Perturbed Label', 'Execution Time', 'F_k', 'Avg Difference', 'Frobenius of Difference']

    # Create csvwriter for each csv file
    with open(targetcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fieldnames)

    # Open ILSVRC label file
    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')
    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    # Define variables for results
    Accuracy = 0
    PerturbedAccuracy = 0
    TargetSuccess = 0
    PerturbedAvgFk = 0
    PerturbedAvgDiff = 0
    PerturbedAvgFroDiff = 0
    counter = 0

    for filename in glob.glob('D:/imageNet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
        im_orig = Image.open(filename).convert('RGB')
        print(filename[47:75])
        if counter == 5000:
            break

        im = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)])(im_orig)

        # returns the 10 nearest labels for the image
        I = targetx_return_I_array(im, net, 10)
        print(I)

        # label to exclude from choice
        exclude_label = I[0]
        # chooses a random int in the I array
        inputNum = random.choice(I)
        if inputNum == exclude_label:
            print("target label same as original label")
            inputNum = random.choice(I)

        print('Generated Eval Label: ', inputNum)

        correct = ILSVRClabels[np.int(counter)].split(' ')[1]
        x = Variable(im.cuda()[None, :], requires_grad=True)
        detect = net.forward(x)
        detected = np.argmax(detect.data.cpu().numpy().flatten())
        str_label_detect = labels[np.int(detected)].split(',')[0]
        print('Detected: ', str_label_detect)

        while (inputNum == I[0] or inputNum == detected):
            inputNum = random.choice(I)
            if inputNum == exclude_label:
                print("target label same as original label")
                inputNum = random.choice(I)
            print('Eval same as correct/detected. Generating new input: ', inputNum)

        str_label_correct = labels[np.int(correct)].split(',')[0]
        str_label_target = labels[np.int(inputNum)].split(',')[0]

        print('Target: ', str_label_target)
        print('Correct: ', str_label_correct)

        start_time = time.time()
        r_tot, loop_i, orig_label, k_i, pert_image, newf_k = targetx_arg(im, net, inputNum, eps)
        end_time = time.time()

        execution_time = end_time - start_time
        print("execution time = " + str(execution_time))

        str_label_orig = labels[np.int(orig_label)].split(',')[0]
        str_label_pert = labels[np.int(k_i)].split(',')[0]

        print("Original label: ", str_label_orig)
        print("Perturbed label: ", str_label_pert)

        if (int(orig_label) == int(correct)):
            print("Classifier is correct")
            Accuracy = Accuracy + 1
        if (int(k_i) == int(correct)):
            print("Classifier is correct on perturbed image")
            PerturbedAccuracy = PerturbedAccuracy + 1
        if (int(k_i) == int(inputNum)):
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
        # Append values to rows, append to csv file
        csvrows.append([filename[47:75], str_label_correct, str_label_orig, str_label_target, str_label_pert,
                        torch.cuda.memory_stats('cuda:0')['active.all.current'], str(loop_i), str(execution_time),
                        newf_k, average, fro])
        with open(targetcsv, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(csvrows)
        print('-------------------------------------------')
        counter = counter + 1

    with open(targetcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(PerturbedAccuracy / 5000)])
        csvwriter.writerows(["Perturbed Target Success: " + str(TargetSuccess / 5000)])
        csvwriter.writerows(["Avg F_k: " + str(PerturbedAvgFk / 5000)])
        csvwriter.writerows(["Avg Difference: " + str(PerturbedAvgDiff / 5000)])
        csvwriter.writerows(["Avg Frobenius of Difference: " + str(PerturbedAvgFroDiff / 5000)])


def runBatchTestTargetFGSM(network, eps, csvname):
    network.eval()
    network.cuda()

    Accuracy = 0
    FGSMAccuracy = 0
    FGSMAvgFk = 0
    FGSMAvgDiff = 0
    FGSMAvgFroDiff = 0
    TargetSuccess = 0

    # Set mean and standard deviation for normalizing image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    fieldnames = ['Image', 'Original Label', 'Classified Label Before Perturbation', 'Perturbed Label',
                  'Memory Usage', 'Time', 'F_k', 'Avg Difference', 'Frobenius of Difference']

    counter = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

    classifier = art.estimators.classification.PyTorchClassifier(
        model=network,
        input_shape=(3, 224, 224),
        loss=criterion,
        optimizer=optimizer,
        nb_classes=1000
    )
    # Create csvwriter for each csv file

    with open(csvname, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    # Open ILSVRC label file
    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')

    for filename in glob.glob(
            'D:/Imagenet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
        print(" \n\n\n**************** T-FGSM *********************\n")
        im_orig = Image.open(filename).convert('RGB')
        print(filename)
        if counter == 5000:
            break
        im = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std)])(im_orig)

        def clip_tensor(A, minv, maxv):
            A = torch.max(A, minv * torch.ones(A.shape))
            A = torch.min(A, maxv * torch.ones(A.shape))
            return A

        clip = lambda x: clip_tensor(x, 0, 255)

        imagetransform = transforms.Compose(
            [transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
             transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
             transforms.Lambda(clip)])

        tensortransform = transforms.Compose([transforms.Scale(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0, 0, 0],
                                                                   std=list(map(lambda x: 1 / x, std))),
                                              transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                              transforms.Lambda(clip)])

        # returns the 10 nearest labels for the image
        # I = targetx_return_I_array(im, network, 10)
        # print(I)
        # I = np.asarray(I)
        I = targetx_return_I_array(im, network, 10)
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

        start_time = time.time()
        input_batch = im.unsqueeze(0)
        result = classifier.predict(input_batch, 1, False)
        label_orig = np.argmax(result.flatten())
        attack = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=eps, norm=np.inf, targeted=True)
        input_array = input_batch.numpy()
        img_adv = attack.generate(x=input_array, y=targetLabel)
        print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
        result_adv = classifier.predict(img_adv, 1, False)
        label_pert = np.argmax(result_adv.flatten())
        end_time = time.time()
        execution_time = end_time - start_time
        print("execution time = " + str(execution_time))

        perturbed_img = img_adv.squeeze()
        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
        correct = ILSVRClabels[np.int(counter)].split(' ')[1]

        str_label_correct = labels[np.int(correct)].split(',')[0]
        str_label_orig = labels[np.int(label_orig)].split(',')[0]
        str_label_pert = labels[np.int(label_pert)].split(',')[0]

        print("Correct label = ", str_label_correct)
        print("Original label = ", str_label_orig)
        print("Perturbed label = ", str_label_pert)
        if (int(label_orig) == int(correct)):
            print("Original classification is correct")
            Accuracy = Accuracy + 1
        pert_image = torch.from_numpy(perturbed_img)

        # If perturbed label matches correct label, add to accuracy count
        if (int(label_pert) == int(correct)):
            print("Classifier is correct")
            FGSMAccuracy = FGSMAccuracy + 1
        if (int(label_pert) != int(I[0])):
            print("Image perturbed to a target label")
            TargetSuccess = TargetSuccess + 1
        diff = imagetransform(pert_image.cpu()) - tensortransform(im_orig)
        fro = np.linalg.norm(diff.numpy())
        average = torch.mean(torch.abs(diff))
        inp = torch.autograd.Variable(torch.from_numpy(input_array[0]).to('cuda:0').float().unsqueeze(0),
                                      requires_grad=True)
        fs = network.forward(inp)
        f_k = (fs[0, label_pert] - fs[0, int(correct)]).data.cpu().numpy()
        FGSMAvgFk = FGSMAvgFk + f_k
        FGSMAvgDiff = FGSMAvgDiff + average
        FGSMAvgFroDiff = FGSMAvgFroDiff + fro
        print(FGSMAvgFk)
        rows = []
        rows.append([filename[47:75], str_label_correct, str_label_orig, str_label_pert,
                     torch.cuda.memory_stats('cuda:0')['active.all.current'], str(execution_time),
                     f_k, average, fro])
        with open(csvname, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(rows)

        print(
            "#################################### END T-FGSM Testing ############################################################\n")
        counter = counter + 1

    # Add total values to csv file

    with open(csvname, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(FGSMAccuracy / 5000)])
        csvwriter.writerows(["Perturbed Target Success: " + str(TargetSuccess / 5000)])
        csvwriter.writerows(["Avg F_k: " + str(FGSMAvgFk / 5000)])
        csvwriter.writerows(["Avg Difference: " + str(FGSMAvgDiff / 5000)])
        csvwriter.writerows(["Avg Frobenius of Difference: " + str(FGSMAvgFroDiff / 5000)])
