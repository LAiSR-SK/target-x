import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets
from PIL import Image
from TargetX import targetx_arg, targetx_return_I_array
from torchvision import models
from models import ResNet, AlexNet, GoogleNet
import os
import time
import random
import glob
from torch.autograd import Variable
import torch
import csv
import art

# these are the needed networks that are to be used
# ILSVRC
res34 = models.resnet34(pretrained=True)
res101 = models.resnet101(pretrained=True)
gnet = models.googlenet(pretrained=True)
anet = models.alexnet(pretrained=True)
vgg = models.vgg19(pretrained=True)
d201 = models.densenet201(pretrained=True)


# CIFAR
# res34 = ResNet.resnet34()
# anet = AlexNet.AlexNet()
# gnet = GoogleNet.GoogLeNet()
#
# stateanet = torch.load("models/alexnet/model.pth")
# anet.load_state_dict(stateanet)
#
# stateres34 = torch.load("models/resnet/model_res34.pth")
# res34.load_state_dict(stateres34)
#
# stategnet = torch.load("models/googlenet/model.pth")
# gnet.load_state_dict(stategnet)


# Define transfer testing function, input network to perturb image with, network to test image, epsilon, and csv name.
def TransferTesting_targetx(net, net2, eps, csvname):
    net.cuda()
    net2.cuda()
    net.eval()
    net2.eval()
    Accuracy = 0
    PerturbedAccuracy = 0
    net2_Accuracy = 0
    net2_PerturbedAccuracy = 0
    TransferableSuccess = 0
    TargetSuccess = 0
    Incorrect_pert_Success = 0
    Net2Correct = 0
    targetx_csv = csvname
    fieldnames = ['Image', 'Correct Label', 'Network 1 Orig Label', 'Network 2 Orig Label', 'Network 1 Pert Label',
                  'Network 2 Pert Label']

    counter = 0

    with open(targetx_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fieldnames)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Load validation data and label set for ILSVRC
    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')
    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    # Loop through 1000 images of validation dataset, choose random label from 0 to 999, launch attack, test, launch attack on second network, test, write to csv.
    for filename in glob.glob('D:/imageNet/ImagenetDataset/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
        if counter == 1000:
            break
        print(" \n\n\n**************** TargetX Approach *********************\n")
        im_orig = Image.open(filename).convert('RGB')
        print(filename)
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
        r, loop_i, label_orig, label_pert, pert_image, newf_k = targetx_arg(im, net, inputNum, eps)
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
        if (int(label_pert) != int(correct) and int(label_pert) != inputNum):
            print("Image perturbed but not to target")
            Incorrect_pert_Success = Incorrect_pert_Success + 1

        label2 = net2(im[None, :].cuda())
        label2 = np.argmax(label2.detach().cpu().numpy())
        str_label_orig2 = labels[np.int(label2)].split(',')[0]
        label_pert2 = net2(pert_image)
        label_pert2 = np.argmax(label_pert2.detach().cpu().numpy())
        str_label_pert2 = labels[np.int(label_pert2)].split(',')[0]

        if (int(label2) == int(correct)):
            print("Net 2 Classifier is correct")
            net2_Accuracy = net2_Accuracy + 1
        if (int(label_pert2) == int(correct)):
            print("Net 2 Classifier is correct on perturbed image")
            net2_PerturbedAccuracy = net2_PerturbedAccuracy + 1
        if (int(label_pert2) == int(label_pert)):
            print("Attack was Transferable")
            TransferableSuccess = TransferableSuccess + 1
        if (int(label_pert2) == int(label2)):
            print("Network 2 Perturbed Label = Network 2 Original Label")
            Net2Correct = Net2Correct + 1

        csvrows = []
        csvrows.append(
            [filename[47:75], str_label_correct, str_label_orig, str_label_orig2, str_label_pert, str_label_pert2])

        with open(csvname, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(csvrows)
        counter = counter + 1

    with open(csvname, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 1000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(PerturbedAccuracy / 1000)])
        csvwriter.writerows(["Transfered Accuracy: " + str(net2_Accuracy / 1000)])
        csvwriter.writerows(["Transferred Perturbed Accuracy: " + str(net2_PerturbedAccuracy / 1000)])
        csvwriter.writerows(["Perturbed To Target Success: " + str(TargetSuccess / 1000)])
        csvwriter.writerows(["Perturbed But Not To Target: " + str(Incorrect_pert_Success / 1000)])
        csvwriter.writerows(["Transferability Success: " + str(TransferableSuccess / 1000)])
        csvwriter.writerows(["Net2 Correctness: " + str(Net2Correct / 1000)])


# Define transfer testing function, input network to perturb image with, network to test image, epsilon, and csv name.
def CIFAR_targetx_TransferTesting(original_net, transfer_net, eps, csvname):
    original_net.cuda()
    transfer_net.cuda()
    original_net.eval()
    transfer_net.eval()

    Accuracy = 0
    PerturbedAccuracy = 0
    net2_Accuracy = 0
    net2_PerturbedAccuracy = 0
    TransferableSuccess = 0
    Incorrect_pert_Success = 0
    TargetSuccess = 0
    Net2Correct = 0

    targetx_csv = csvname

    fieldnames = ['Image', 'Correct Label', 'Network 1 Orig Label', 'Network 2 Orig Label', 'Network 1 Pert Label',
                  'Network 2 Pert Label']

    with open(targetx_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fieldnames)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose([transforms.ToTensor()])

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=False, transform=transform)
    # Loop through 1000 images of CIFAR-10 test dataset, choose random label from 0 to 9, launch attack, test, launch attack on second network, test, write to csv.
    counter = 0
    for i, data in enumerate(testset):  # assuming jpg
        inputs, labels = data
        if counter == 1000:
            break
        print(" \n\n\n**************** TargetX Approach *********************\n")

        inputNum = random.randint(0, 9)
        print('Generated Eval Label: ', inputNum)
        correct = labels
        detect = original_net.forward(inputs[None, ...].cuda())
        detected = np.argmax(detect.data.cpu().numpy().flatten())
        str_label_detect = classes[np.int(detected)].split(',')[0]

        print('Detected: ', str_label_detect)

        while (inputNum == int(correct) or inputNum == detected):
            inputNum = random.randint(0, 9)
            print('Eval same as correct/detected. Generating new input: ', inputNum)

        str_label_correct = classes[np.int(correct)].split(',')[0]
        str_label_target = classes[np.int(inputNum)].split(',')[0]

        print('Target: ', str_label_target)
        print('Correct: ', str_label_correct)

        start_time = time.time()
        r, loop_i, label_orig, label_pert, pert_image, newf_k = targetx_arg(inputs, original_net, inputNum, eps)
        end_time = time.time()

        execution_time = end_time - start_time
        print("execution time = " + str(execution_time))

        str_label_orig = classes[np.int(label_orig)].split(',')[0]
        str_label_pert = classes[np.int(label_pert)].split(',')[0]

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
        if (int(label_pert) != int(correct) and int(label_pert) != inputNum):
            print("Image perturbed but not to target")
            Incorrect_pert_Success = Incorrect_pert_Success + 1

        label2 = transfer_net(inputs[None, :].cuda())
        label2 = np.argmax(label2.detach().cpu().numpy())
        str_label_orig2 = classes[np.int(label2)].split(',')[0]
        label_pert2 = transfer_net(pert_image)
        label_pert2 = np.argmax(label_pert2.detach().cpu().numpy())
        str_label_pert2 = classes[np.int(label_pert2)].split(',')[0]

        if (int(label2) == int(correct)):
            print("Net 2 Classifier is correct")
            net2_Accuracy = net2_Accuracy + 1
        if (int(label_pert2) == int(correct)):
            print("Net 2 Classifier is correct on perturbed image")
            net2_PerturbedAccuracy = net2_PerturbedAccuracy + 1
        if (int(label_pert2) == int(label_pert)):
            print("Attack was Transferable")
            TransferableSuccess = TransferableSuccess + 1
        if (int(label_pert2) == int(label2)):
            print("Network 2 Perturbed Label = Network 2 Original Label")
            Net2Correct = Net2Correct + 1

        csvrows = []
        csvrows.append(
            [i, str_label_correct, str_label_orig, str_label_orig2, str_label_pert, str_label_pert2])

        with open(csvname, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(csvrows)
        counter = counter + 1

    with open(csvname, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 1000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(PerturbedAccuracy / 1000)])
        csvwriter.writerows(["Transfered Accuracy: " + str(net2_Accuracy / 1000)])
        csvwriter.writerows(["Transferred Perturbed Accuracy: " + str(net2_PerturbedAccuracy / 1000)])
        csvwriter.writerows(["Perturbed Target Success: " + str(TargetSuccess / 1000)])
        csvwriter.writerows(["Perturbed But Not To Target: " + str(Incorrect_pert_Success / 1000)])
        csvwriter.writerows(["Transferability Success: " + str(TransferableSuccess / 1000)])
        csvwriter.writerows(["Net2 Correctness: " + str(Net2Correct / 1000)])


def TransferTesting_TargetFGSM(net, net2, eps, csvname):
    net.cuda()
    net2.cuda()
    net.eval()
    net2.eval()
    Accuracy = 0
    PerturbedAccuracy = 0
    net2_Accuracy = 0
    net2_PerturbedAccuracy = 0
    TransferableSuccess = 0
    TargetSuccess = 0
    Incorrect_pert_Success = 0
    Net2Correct = 0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    fgsmcsv = csvname

    fieldnames = ['Image', 'Correct Label', 'Network 1 Orig Label', 'Network 2 Orig Label', 'Network 1 Pert Label',
                  'Network 2 Pert Label']

    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')
    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    eps = eps
    counter = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    classifier = art.estimators.classification.PyTorchClassifier(
        model=net,
        input_shape=(3, 224, 224),
        loss=criterion,
        optimizer=optimizer,
        nb_classes=1000
    )

    with open(fgsmcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fieldnames)

    for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
        if counter == 5000:
            break
        print('T-FGSM Testing')
        im_orig = Image.open(filename).convert('RGB')
        print(filename)
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
        I = I - exclude_label

        # deteted image
        x = Variable(im.cuda()[None, :], requires_grad=True)
        detect = net.forward(x)
        detected = np.argmax(detect.data.cpu().numpy().flatten())
        str_label_detect = labels[np.int(detected)].split(',')[0]
        print('Detected: ', str_label_detect)

        start_time = time.time()
        input_batch = im.unsqueeze(0)
        result = classifier.predict(input_batch, 1, False)
        label_orig = np.argmax(result.flatten())
        attack = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=eps, norm=np.inf, targeted=True)
        input_array = input_batch.numpy()
        img_adv = attack.generate(x=input_array, y=I)
        print("Memory Usage: ", torch.cuda.memory_stats('cuda:0')['active.all.current'])
        result_adv = classifier.predict(img_adv, 1, False)
        label_pert = np.argmax(result_adv.flatten())

        end_time = time.time()
        execution_time = end_time - start_time
        print("execution time = " + str(execution_time))

        pert_img = img_adv.squeeze()
        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
        correct = ILSVRClabels[np.int(counter)].split(' ')[1]

        str_label_correct = labels[np.int(correct)].split(',')[0]
        str_label_orig = labels[np.int(label_orig)].split(',')[0]
        str_label_pert = labels[np.int(label_pert)].split(',')[0]

        print("Correct label = ", str_label_correct)
        print("Original label = ", str_label_orig)
        print("Perturbed label = ", str_label_pert)

        if (int(label_orig) == int(correct)):
            print("Classifier is correct")
            Accuracy = Accuracy + 1
        if (int(label_pert) == int(correct)):
            print("Classifier is correct on perturbed image")
            PerturbedAccuracy = PerturbedAccuracy + 1
        if (int(label_pert) != int(I[0])):
            print("Image perturbed to a target")
            TargetSuccess = TargetSuccess + 1
        if (int(label_pert) != int(correct) and int(label_pert) not in I):
            print("Image perturbed but not to target")
            Incorrect_pert_Success = Incorrect_pert_Success + 1

        label2 = net2(im[None, :].cuda())
        label2 = np.argmax(label2.detach().cpu().numpy())
        str_label_orig2 = labels[np.int(label2)].split(',')[0]
        label_pert2 = net2(pert_img)
        label_pert2 = np.argmax(label_pert2.detach().cpu().numpy())
        str_label_pert2 = labels[np.int(label_pert2)].split(',')[0]

        if (int(label2) == int(correct)):
            print("Net 2 Classifier is correct")
            net2_Accuracy = net2_Accuracy + 1
        if (int(label_pert2) == int(correct)):
            print("Net 2 Classifier is correct on perturbed image")
            net2_PerturbedAccuracy = net2_PerturbedAccuracy + 1
        if (int(label_pert2) == int(label_pert)):
            print("Attack was Transferable")
            TransferableSuccess = TransferableSuccess + 1
        if (int(label_pert2) == int(label2)):
            print("Network 2 Perturbed Label = Network 2 Original Label")
            Net2Correct = Net2Correct + 1

        fgsmrows = []
        fgsmrows.append(
            [filename[47:75], str_label_correct, str_label_orig, str_label_orig2, str_label_pert, str_label_pert2])

        with open(fgsmcsv, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(fgsmrows)
        counter = counter + 1

    with open(fgsmcsv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Original Accuracy: " + str(Accuracy / 5000)])
        csvwriter.writerows(["Original Perturbed Accuracy: " + str(PerturbedAccuracy / 5000)])
        csvwriter.writerows(["Transfered Accuracy: " + str(net2_Accuracy / 5000)])
        csvwriter.writerows(["Transfered Perturbed Accuracy: " + str(net2_PerturbedAccuracy / 5000)])
        csvwriter.writerows(["Perturbed To Target Success: " + str(TargetSuccess / 5000)])
        csvwriter.writerows(["Perturbed But Not To Target: " + str(Incorrect_pert_Success / 5000)])
        csvwriter.writerows(["Transferability Success: " + str(TransferableSuccess / 5000)])
        csvwriter.writerows(["Net2 Correctness: " + str(Net2Correct / 5000)])


# AlexNet
TransferTesting_TargetFGSM(anet, gnet, 0.0005, 'transfer_TargetFGSM_AlexNet_to_GoogleNet_0.0005.csv')
# TransferTesting_TargetFGSM(anet, gnet, 0.0001, 'transfer_TargetFGSM_AlexNet_to_GoogleNet_0.0001.csv')
# TransferTesting_TargetFGSM(anet, gnet, 0.005, 'transfer_TargetFGSM_AlexNet_to_GoogleNet_0.005.csv')
# TransferTesting_TargetFGSM(anet, gnet, 0.001, 'transfer_TargetFGSM_AlexNet_to_GoogleNet_0.001.csv')
# TransferTesting_TargetFGSM(anet, gnet, 0.05, 'transfer_TargetFGSM_AlexNet_to_GoogleNet_0.05.csv')
# TransferTesting_TargetFGSM(anet, gnet, 0.01, 'transfer_TargetFGSM_AlexNet_to_GoogleNet_0.01.csv')
# TransferTesting_TargetFGSM(anet, gnet, 0.1, 'transfer_TargetFGSM_AlexNet_to_GoogleNet_0.1.csv')
# TransferTesting_TargetFGSM(anet, gnet, 0.2, 'transfer_TargetFGSM_AlexNet_to_GoogleNet_0.2.csv')
#
# TransferTesting_TargetFGSM(anet, res34, 0.0005, 'transfer_TargetFGSM_AlexNet_to_ResNet34_0.0005.csv')
# TransferTesting_TargetFGSM(anet, res34, 0.0001, 'transfer_TargetFGSM_AlexNet_to_ResNet34_0.0001.csv')
# TransferTesting_TargetFGSM(anet, res34, 0.005, 'transfer_TargetFGSM_AlexNet_to_ResNet34_0.005.csv')
# TransferTesting_TargetFGSM(anet, res34, 0.001, 'transfer_TargetFGSM_AlexNet_to_ResNet34_0.001.csv')
# TransferTesting_TargetFGSM(anet, res34, 0.05, 'transfer_TargetFGSM_AlexNet_to_ResNet34_0.05.csv')
# TransferTesting_TargetFGSM(anet, res34, 0.01, 'transfer_TargetFGSM_AlexNet_to_ResNet34_0.01.csv')
# TransferTesting_TargetFGSM(anet, res34, 0.1, 'transfer_TargetFGSM_AlexNet_to_ResNet34_0.1.csv')
# TransferTesting_TargetFGSM(anet, res34, 0.2, 'transfer_TargetFGSM_AlexNet_to_ResNet34_0.2.csv')
#
# TransferTesting_TargetFGSM(anet, res101, 0.0005, 'transfer_TargetFGSM_AlexNet_to_ResNet101_0.0005.csv')
# TransferTesting_TargetFGSM(anet, res101, 0.0001, 'transfer_TargetFGSM_AlexNet_to_ResNet101_0.0001.csv')
# TransferTesting_TargetFGSM(anet, res101, 0.005, 'transfer_TargetFGSM_AlexNet_to_ResNet101_0.005.csv')
# TransferTesting_TargetFGSM(anet, res101, 0.001, 'transfer_TargetFGSM_AlexNet_to_ResNet101_0.001.csv')
# TransferTesting_TargetFGSM(anet, res101, 0.05, 'transfer_TargetFGSM_AlexNet_to_ResNet101_0.05.csv')
# TransferTesting_TargetFGSM(anet, res101, 0.01, 'transfer_TargetFGSM_AlexNet_to_ResNet101_0.01.csv')
# TransferTesting_TargetFGSM(anet, res101, 0.1, 'transfer_TargetFGSM_AlexNet_to_ResNet101_0.1.csv')
# TransferTesting_TargetFGSM(anet, res101, 0.2, 'transfer_TargetFGSM_AlexNet_to_ResNet101_0.2.csv')
#
# TransferTesting_TargetFGSM(anet, vgg, 0.0005, 'transfer_TargetFGSM_AlexNet_to_VGG19_0.0005.csv')
# TransferTesting_TargetFGSM(anet, vgg, 0.0001, 'transfer_TargetFGSM_AlexNet_to_VGG19_0.0001.csv')
# TransferTesting_TargetFGSM(anet, vgg, 0.005, 'transfer_TargetFGSM_AlexNet_to_VGG19_0.005.csv')
# TransferTesting_TargetFGSM(anet, vgg, 0.001, 'transfer_TargetFGSM_AlexNet_to_VGG19_0.001.csv')
# TransferTesting_TargetFGSM(anet, vgg, 0.05, 'transfer_TargetFGSM_AlexNet_to_VGG19_0.05.csv')
# TransferTesting_TargetFGSM(anet, vgg, 0.01, 'transfer_TargetFGSM_AlexNet_to_VGG19_0.01.csv')
# TransferTesting_TargetFGSM(anet, vgg, 0.1, 'transfer_TargetFGSM_AlexNet_to_VGG19_0.1.csv')
# TransferTesting_TargetFGSM(anet, vgg, 0.2, 'transfer_TargetFGSM_AlexNet_to_VGG19_0.2.csv')
#
# TransferTesting_TargetFGSM(anet, d201, 0.0005, 'transfer_TargetFGSM_AlexNet_to_DenseNet201_0.0005.csv')
# TransferTesting_TargetFGSM(anet, d201, 0.0001, 'transfer_TargetFGSM_AlexNet_to_DenseNet201_0.0001.csv')
# TransferTesting_TargetFGSM(anet, d201, 0.005, 'transfer_TargetFGSM_AlexNet_to_DenseNet201_0.005.csv')
# TransferTesting_TargetFGSM(anet, d201, 0.001, 'transfer_TargetFGSM_AlexNet_to_DenseNet201_0.001.csv')
# TransferTesting_TargetFGSM(anet, d201, 0.05, 'transfer_TargetFGSM_AlexNet_to_DenseNet201_0.05.csv')
# TransferTesting_TargetFGSM(anet, d201, 0.01, 'transfer_TargetFGSM_AlexNet_to_DenseNet201_0.01.csv')
# TransferTesting_TargetFGSM(anet, d201, 0.1, 'transfer_TargetFGSM_AlexNet_to_DenseNet201_0.1.csv')
# TransferTesting_TargetFGSM(anet, d201, 0.2, 'transfer_TargetFGSM_AlexNet_to_DenseNet201_0.2.csv')
#
# # GoogleNet
# TransferTesting_TargetFGSM(gnet, anet, 0.0005, 'transfer_TargetFGSM_GoogleNet_to_AlexNet_0.0005.csv')
# TransferTesting_TargetFGSM(gnet, anet, 0.0001, 'transfer_TargetFGSM_GoogleNet_to_AlexNet_0.0001.csv')
# TransferTesting_TargetFGSM(gnet, anet, 0.005, 'transfer_TargetFGSM_GoogleNet_to_AlexNet_0.005.csv')
# TransferTesting_TargetFGSM(gnet, anet, 0.001, 'transfer_TargetFGSM_GoogleNet_to_AlexNet_0.001.csv')
# TransferTesting_TargetFGSM(gnet, anet, 0.05, 'transfer_TargetFGSM_GoogleNet_to_AlexNet_0.05.csv')
# TransferTesting_TargetFGSM(gnet, anet, 0.01, 'transfer_TargetFGSM_GoogleNet_to_AlexNet_0.01.csv')
# TransferTesting_TargetFGSM(gnet, anet, 0.1, 'transfer_TargetFGSM_GoogleNet_to_AlexNet_0.1.csv')
# TransferTesting_TargetFGSM(gnet, anet, 0.2, 'transfer_TargetFGSM_GoogleNet_to_AlexNet_0.2.csv')
#
# TransferTesting_TargetFGSM(gnet, res34, 0.0005, 'transfer_TargetFGSM_GoogleNet_to_ResNet34_0.0005.csv')
# TransferTesting_TargetFGSM(gnet, res34, 0.0001, 'transfer_TargetFGSM_GoogleNet_to_ResNet34_0.0001.csv')
# TransferTesting_TargetFGSM(gnet, res34, 0.005, 'transfer_TargetFGSM_GoogleNet_to_ResNet34_0.005.csv')
# TransferTesting_TargetFGSM(gnet, res34, 0.001, 'transfer_TargetFGSM_GoogleNet_to_ResNet34_0.001.csv')
# TransferTesting_TargetFGSM(gnet, res34, 0.05, 'transfer_TargetFGSM_GoogleNet_to_ResNet34_0.05.csv')
# TransferTesting_TargetFGSM(gnet, res34, 0.01, 'transfer_TargetFGSM_GoogleNet_to_ResNet34_0.01.csv')
# TransferTesting_TargetFGSM(gnet, res34, 0.1, 'transfer_TargetFGSM_GoogleNet_to_ResNet34_0.1.csv')
# TransferTesting_TargetFGSM(gnet, res34, 0.2, 'transfer_TargetFGSM_GoogleNet_to_ResNet34_0.2.csv')
#
# TransferTesting_TargetFGSM(gnet, res101, 0.0005, 'transfer_TargetFGSM_GoogleNet_to_ResNet101_0.0005.csv')
# TransferTesting_TargetFGSM(gnet, res101, 0.0001, 'transfer_TargetFGSM_GoogleNet_to_ResNet101_0.0001.csv')
# TransferTesting_TargetFGSM(gnet, res101, 0.005, 'transfer_TargetFGSM_GoogleNet_to_ResNet101_0.005.csv')
# TransferTesting_TargetFGSM(gnet, res101, 0.001, 'transfer_TargetFGSM_GoogleNet_to_ResNet101_0.001.csv')
# TransferTesting_TargetFGSM(gnet, res101, 0.05, 'transfer_TargetFGSM_GoogleNet_to_ResNet101_0.05.csv')
# TransferTesting_TargetFGSM(gnet, res101, 0.01, 'transfer_TargetFGSM_GoogleNet_to_ResNet101_0.01.csv')
# TransferTesting_TargetFGSM(gnet, res101, 0.1, 'transfer_TargetFGSM_GoogleNet_to_ResNet101_0.1.csv')
# TransferTesting_TargetFGSM(gnet, res101, 0.2, 'transfer_TargetFGSM_GoogleNet_to_ResNet101_0.2.csv')
#
# TransferTesting_TargetFGSM(gnet, vgg, 0.0005, 'transfer_TargetFGSM_GoogleNet_to_VGG19_0.0005.csv')
# TransferTesting_TargetFGSM(gnet, vgg, 0.0001, 'transfer_TargetFGSM_GoogleNet_to_VGG19_0.0001.csv')
# TransferTesting_TargetFGSM(gnet, vgg, 0.005, 'transfer_TargetFGSM_GoogleNet_to_VGG19_0.005.csv')
# TransferTesting_TargetFGSM(gnet, vgg, 0.001, 'transfer_TargetFGSM_GoogleNet_to_VGG19_0.001.csv')
# TransferTesting_TargetFGSM(gnet, vgg, 0.05, 'transfer_TargetFGSM_GoogleNet_to_VGG19_0.05.csv')
# TransferTesting_TargetFGSM(gnet, vgg, 0.01, 'transfer_TargetFGSM_GoogleNet_to_VGG19_0.01.csv')
# TransferTesting_TargetFGSM(gnet, vgg, 0.1, 'transfer_TargetFGSM_GoogleNet_to_VGG19_0.1.csv')
# TransferTesting_TargetFGSM(gnet, vgg, 0.2, 'transfer_TargetFGSM_GoogleNet_to_VGG19_0.2.csv')
#
# TransferTesting_TargetFGSM(gnet, d201, 0.0005, 'transfer_TargetFGSM_GoogleNet_to_DenseNet201_0.0005.csv')
# TransferTesting_TargetFGSM(gnet, d201, 0.0001, 'transfer_TargetFGSM_GoogleNet_to_DenseNet201_0.0001.csv')
# TransferTesting_TargetFGSM(gnet, d201, 0.005, 'transfer_TargetFGSM_GoogleNet_to_DenseNet201_0.005.csv')
# TransferTesting_TargetFGSM(gnet, d201, 0.001, 'transfer_TargetFGSM_GoogleNet_to_DenseNet201_0.001.csv')
# TransferTesting_TargetFGSM(gnet, d201, 0.05, 'transfer_TargetFGSM_GoogleNet_to_DenseNet201_0.05.csv')
# TransferTesting_TargetFGSM(gnet, d201, 0.01, 'transfer_TargetFGSM_GoogleNet_to_DenseNet201_0.01.csv')
# TransferTesting_TargetFGSM(gnet, d201, 0.1, 'transfer_TargetFGSM_GoogleNet_to_DenseNet201_0.1.csv')
# TransferTesting_TargetFGSM(gnet, d201, 0.2, 'transfer_TargetFGSM_GoogleNet_to_DenseNet201_0.2.csv')
#
# # ResNet-34
# TransferTesting_TargetFGSM(res34, anet, 0.0005, 'transfer_TargetFGSM_ResNet34_to_AlexNet_0.0005.csv')
# TransferTesting_TargetFGSM(res34, anet, 0.0001, 'transfer_TargetFGSM_ResNet34_to_AlexNet_0.0001.csv')
# TransferTesting_TargetFGSM(res34, anet, 0.005, 'transfer_TargetFGSM_ResNet34_to_AlexNet_0.005.csv')
# TransferTesting_TargetFGSM(res34, anet, 0.001, 'transfer_TargetFGSM_ResNet34_to_AlexNet_0.001.csv')
# TransferTesting_TargetFGSM(res34, anet, 0.05, 'transfer_TargetFGSM_ResNet34_to_AlexNet_0.05.csv')
# TransferTesting_TargetFGSM(res34, anet, 0.01, 'transfer_TargetFGSM_ResNet34_to_AlexNet_0.01.csv')
# TransferTesting_TargetFGSM(res34, anet, 0.1, 'transfer_TargetFGSM_ResNet34_to_AlexNet_0.1.csv')
# TransferTesting_TargetFGSM(res34, anet, 0.2, 'transfer_TargetFGSM_ResNet34_to_AlexNet_0.2.csv')
#
# TransferTesting_TargetFGSM(res34, gnet, 0.0005, 'transfer_TargetFGSM_ResNet34_to_GoogleNet_0.0005.csv')
# TransferTesting_TargetFGSM(res34, gnet, 0.0001, 'transfer_TargetFGSM_ResNet34_to_GoogleNet_0.0001.csv')
# TransferTesting_TargetFGSM(res34, gnet, 0.005, 'transfer_TargetFGSM_ResNet34_to_GoogleNet_0.005.csv')
# TransferTesting_TargetFGSM(res34, gnet, 0.001, 'transfer_TargetFGSM_ResNet34_to_GoogleNet_0.001.csv')
# TransferTesting_TargetFGSM(res34, gnet, 0.05, 'transfer_TargetFGSM_ResNet34_to_GoogleNet_0.05.csv')
# TransferTesting_TargetFGSM(res34, gnet, 0.01, 'transfer_TargetFGSM_ResNet34_to_GoogleNet_0.01.csv')
# TransferTesting_TargetFGSM(res34, gnet, 0.1, 'transfer_TargetFGSM_ResNet34_to_GoogleNet_0.1.csv')
# TransferTesting_TargetFGSM(res34, gnet, 0.2, 'transfer_TargetFGSM_ResNet34_to_GoogleNet_0.2.csv')
#
# TransferTesting_TargetFGSM(res34, res101, 0.0005, 'transfer_TargetFGSM_ResNet34_to_ResNet101_0.0005.csv')
# TransferTesting_TargetFGSM(res34, res101, 0.0001, 'transfer_TargetFGSM_ResNet34_to_ResNet101_0.0001.csv')
# TransferTesting_TargetFGSM(res34, res101, 0.005, 'transfer_TargetFGSM_ResNet34_to_ResNet101_0.005.csv')
# TransferTesting_TargetFGSM(res34, res101, 0.001, 'transfer_TargetFGSM_ResNet34_to_ResNet101_0.001.csv')
# TransferTesting_TargetFGSM(res34, res101, 0.05, 'transfer_TargetFGSM_ResNet34_to_ResNet101_0.05.csv')
# TransferTesting_TargetFGSM(res34, res101, 0.01, 'transfer_TargetFGSM_ResNet34_to_ResNet101_0.01.csv')
# TransferTesting_TargetFGSM(res34, res101, 0.1, 'transfer_TargetFGSM_ResNet34_to_ResNet101_0.1.csv')
# TransferTesting_TargetFGSM(res34, res101, 0.2, 'transfer_TargetFGSM_ResNet34_to_ResNet101_0.2.csv')
#
# TransferTesting_TargetFGSM(res34, vgg, 0.0005, 'transfer_TargetFGSM_ResNet34_to_VGG19_0.0005.csv')
# TransferTesting_TargetFGSM(res34, vgg, 0.0001, 'transfer_TargetFGSM_ResNet34_to_VGG19_0.0001.csv')
# TransferTesting_TargetFGSM(res34, vgg, 0.005, 'transfer_TargetFGSM_ResNet34_to_VGG19_0.005.csv')
# TransferTesting_TargetFGSM(res34, vgg, 0.001, 'transfer_TargetFGSM_ResNet34_to_VGG19_0.001.csv')
# TransferTesting_TargetFGSM(res34, vgg, 0.05, 'transfer_TargetFGSM_ResNet34_to_VGG19_0.05.csv')
# TransferTesting_TargetFGSM(res34, vgg, 0.01, 'transfer_TargetFGSM_ResNet34_to_VGG19_0.01.csv')
# TransferTesting_TargetFGSM(res34, vgg, 0.1, 'transfer_TargetFGSM_ResNet34_to_VGG19_0.1.csv')
# TransferTesting_TargetFGSM(res34, vgg, 0.2, 'transfer_TargetFGSM_ResNet34_to_VGG19_0.2.csv')
#
# TransferTesting_TargetFGSM(res34, d201, 0.0005, 'transfer_TargetFGSM_ResNet34_to_DenseNet201_0.0005.csv')
# TransferTesting_TargetFGSM(res34, d201, 0.0001, 'transfer_TargetFGSM_ResNet34_to_DenseNet201_0.0001.csv')
# TransferTesting_TargetFGSM(res34, d201, 0.005, 'transfer_TargetFGSM_ResNet34_to_DenseNet201_0.005.csv')
# TransferTesting_TargetFGSM(res34, d201, 0.001, 'transfer_TargetFGSM_ResNet34_to_DenseNet201_0.001.csv')
# TransferTesting_TargetFGSM(res34, d201, 0.05, 'transfer_TargetFGSM_ResNet34_to_DenseNet201_0.05.csv')
# TransferTesting_TargetFGSM(res34, d201, 0.01, 'transfer_TargetFGSM_ResNet34_to_DenseNet201_0.01.csv')
# TransferTesting_TargetFGSM(res34, d201, 0.1, 'transfer_TargetFGSM_ResNet34_to_DenseNet201_0.1.csv')
# TransferTesting_TargetFGSM(res34, d201, 0.2, 'transfer_TargetFGSM_ResNet34_to_DenseNet201_0.2.csv')
#
# # ResNet-101
# TransferTesting_TargetFGSM(res101, anet, 0.0005, 'transfer_TargetFGSM_ResNet101_to_AlexNet_0.0005.csv')
# TransferTesting_TargetFGSM(res101, anet, 0.0001, 'transfer_TargetFGSM_ResNet101_to_AlexNet_0.0001.csv')
# TransferTesting_TargetFGSM(res101, anet, 0.005, 'transfer_TargetFGSM_ResNet101_to_AlexNet_0.005.csv')
# TransferTesting_TargetFGSM(res101, anet, 0.001, 'transfer_TargetFGSM_ResNet101_to_AlexNet_0.001.csv')
# TransferTesting_TargetFGSM(res101, anet, 0.05, 'transfer_TargetFGSM_ResNet101_to_AlexNet_0.05.csv')
# TransferTesting_TargetFGSM(res101, anet, 0.01, 'transfer_TargetFGSM_ResNet101_to_AlexNet_0.01.csv')
# TransferTesting_TargetFGSM(res101, anet, 0.1, 'transfer_TargetFGSM_ResNet101_to_AlexNet_0.1.csv')
# TransferTesting_TargetFGSM(res101, anet, 0.2, 'transfer_TargetFGSM_ResNet101_to_AlexNet_0.2.csv')
#
# TransferTesting_TargetFGSM(res101, gnet, 0.0005, 'transfer_TargetFGSM_ResNet101_to_GoogleNet_0.0005.csv')
# TransferTesting_TargetFGSM(res101, gnet, 0.0001, 'transfer_TargetFGSM_ResNet101_to_GoogleNet_0.0001.csv')
# TransferTesting_TargetFGSM(res101, gnet, 0.005, 'transfer_TargetFGSM_ResNet101_to_GoogleNet_0.005.csv')
# TransferTesting_TargetFGSM(res101, gnet, 0.001, 'transfer_TargetFGSM_ResNet101_to_GoogleNet_0.001.csv')
# TransferTesting_TargetFGSM(res101, gnet, 0.05, 'transfer_TargetFGSM_ResNet101_to_GoogleNet_0.05.csv')
# TransferTesting_TargetFGSM(res101, gnet, 0.01, 'transfer_TargetFGSM_ResNet101_to_GoogleNet_0.01.csv')
# TransferTesting_TargetFGSM(res101, gnet, 0.1, 'transfer_TargetFGSM_ResNet101_to_GoogleNet_0.1.csv')
# TransferTesting_TargetFGSM(res101, gnet, 0.2, 'transfer_TargetFGSM_ResNet101_to_GoogleNet_0.2.csv')
#
# TransferTesting_TargetFGSM(res101, res34, 0.0005, 'transfer_TargetFGSM_ResNet101_to_ResNet34_0.0005.csv')
# TransferTesting_TargetFGSM(res101, res34, 0.0001, 'transfer_TargetFGSM_ResNet101_to_ResNet34_0.0001.csv')
# TransferTesting_TargetFGSM(res101, res34, 0.005, 'transfer_TargetFGSM_ResNet101_to_ResNet34_0.005.csv')
# TransferTesting_TargetFGSM(res101, res34, 0.001, 'transfer_TargetFGSM_ResNet101_to_ResNet34_0.001.csv')
# TransferTesting_TargetFGSM(res101, res34, 0.05, 'transfer_TargetFGSM_ResNet101_to_ResNet34_0.05.csv')
# TransferTesting_TargetFGSM(res101, res34, 0.01, 'transfer_TargetFGSM_ResNet101_to_ResNet34_0.01.csv')
# TransferTesting_TargetFGSM(res101, res34, 0.1, 'transfer_TargetFGSM_ResNet101_to_ResNet34_0.1.csv')
# TransferTesting_TargetFGSM(res101, res34, 0.2, 'transfer_TargetFGSM_ResNet101_to_ResNet34_0.2.csv')
#
# TransferTesting_TargetFGSM(res101, vgg, 0.0005, 'transfer_TargetFGSM_ResNet101_to_VGG19_0.0005.csv')
# TransferTesting_TargetFGSM(res101, vgg, 0.0001, 'transfer_TargetFGSM_ResNet101_to_VGG19_0.0001.csv')
# TransferTesting_TargetFGSM(res101, vgg, 0.005, 'transfer_TargetFGSM_ResNet101_to_VGG19_0.005.csv')
# TransferTesting_TargetFGSM(res101, vgg, 0.001, 'transfer_TargetFGSM_ResNet101_to_VGG19_0.001.csv')
# TransferTesting_TargetFGSM(res101, vgg, 0.05, 'transfer_TargetFGSM_ResNet101_to_VGG19_0.05.csv')
# TransferTesting_TargetFGSM(res101, vgg, 0.01, 'transfer_TargetFGSM_ResNet101_to_VGG19_0.01.csv')
# TransferTesting_TargetFGSM(res101, vgg, 0.1, 'transfer_TargetFGSM_ResNet101_to_VGG19_0.1.csv')
# TransferTesting_TargetFGSM(res101, vgg, 0.2, 'transfer_TargetFGSM_ResNet101_to_VGG19_0.2.csv')
#
# TransferTesting_TargetFGSM(res101, d201, 0.0005, 'transfer_TargetFGSM_ResNet101_to_DenseNet201_0.0005.csv')
# TransferTesting_TargetFGSM(res101, d201, 0.0001, 'transfer_TargetFGSM_ResNet101_to_DenseNet201_0.0001.csv')
# TransferTesting_TargetFGSM(res101, d201, 0.005, 'transfer_TargetFGSM_ResNet101_to_DenseNet201_0.005.csv')
# TransferTesting_TargetFGSM(res101, d201, 0.001, 'transfer_TargetFGSM_ResNet101_to_DenseNet201_0.001.csv')
# TransferTesting_TargetFGSM(res101, d201, 0.05, 'transfer_TargetFGSM_ResNet101_to_DenseNet201_0.05.csv')
# TransferTesting_TargetFGSM(res101, d201, 0.01, 'transfer_TargetFGSM_ResNet101_to_DenseNet201_0.01.csv')
# TransferTesting_TargetFGSM(res101, d201, 0.1, 'transfer_TargetFGSM_ResNet101_to_DenseNet201_0.1.csv')
# TransferTesting_TargetFGSM(res101, d201, 0.2, 'transfer_TargetFGSM_ResNet101_to_DenseNet201_0.2.csv')
#
# # VGG-19
# TransferTesting_TargetFGSM(vgg, anet, 0.0005, 'transfer_TargetFGSM_VGG19_to_AlexNet_0.0005.csv')
# TransferTesting_TargetFGSM(vgg, anet, 0.0001, 'transfer_TargetFGSM_VGG19_to_AlexNet_0.0001.csv')
# TransferTesting_TargetFGSM(vgg, anet, 0.005, 'transfer_TargetFGSM_VGG19_to_AlexNet_0.005.csv')
# TransferTesting_TargetFGSM(vgg, anet, 0.001, 'transfer_TargetFGSM_VGG19_to_AlexNet_0.001.csv')
# TransferTesting_TargetFGSM(vgg, anet, 0.05, 'transfer_TargetFGSM_VGG19_to_AlexNet_0.05.csv')
# TransferTesting_TargetFGSM(vgg, anet, 0.01, 'transfer_TargetFGSM_VGG19_to_AlexNet_0.01.csv')
# TransferTesting_TargetFGSM(vgg, anet, 0.1, 'transfer_TargetFGSM_VGG19_to_AlexNet_0.1.csv')
# TransferTesting_TargetFGSM(vgg, anet, 0.2, 'transfer_TargetFGSM_VGG19_to_AlexNet_0.2.csv')
#
# TransferTesting_TargetFGSM(vgg, gnet, 0.0005, 'transfer_TargetFGSM_VGG19_to_GoogleNet_0.0005.csv')
# TransferTesting_TargetFGSM(vgg, gnet, 0.0001, 'transfer_TargetFGSM_VGG19_to_GoogleNet_0.0001.csv')
# TransferTesting_TargetFGSM(vgg, gnet, 0.005, 'transfer_TargetFGSM_VGG19_to_GoogleNet_0.005.csv')
# TransferTesting_TargetFGSM(vgg, gnet, 0.001, 'transfer_TargetFGSM_VGG19_to_GoogleNet_0.001.csv')
# TransferTesting_TargetFGSM(vgg, gnet, 0.05, 'transfer_TargetFGSM_VGG19_to_GoogleNet_0.05.csv')
# TransferTesting_TargetFGSM(vgg, gnet, 0.01, 'transfer_TargetFGSM_VGG19_to_GoogleNet_0.01.csv')
# TransferTesting_TargetFGSM(vgg, gnet, 0.1, 'transfer_TargetFGSM_VGG19_to_GoogleNet_0.1.csv')
# TransferTesting_TargetFGSM(vgg, gnet, 0.2, 'transfer_TargetFGSM_VGG19_to_GoogleNet_0.2.csv')
#
# TransferTesting_TargetFGSM(vgg, res34, 0.0005, 'transfer_TargetFGSM_VGG19_to_ResNet34_0.0005.csv')
# TransferTesting_TargetFGSM(vgg, res34, 0.0001, 'transfer_TargetFGSM_VGG19_to_ResNet34_0.0001.csv')
# TransferTesting_TargetFGSM(vgg, res34, 0.005, 'transfer_TargetFGSM_VGG19_to_ResNet34_0.005.csv')
# TransferTesting_TargetFGSM(vgg, res34, 0.001, 'transfer_TargetFGSM_VGG19_to_ResNet34_0.001.csv')
# TransferTesting_TargetFGSM(vgg, res34, 0.05, 'transfer_TargetFGSM_VGG19_to_ResNet34_0.05.csv')
# TransferTesting_TargetFGSM(vgg, res34, 0.01, 'transfer_TargetFGSM_VGG19_to_ResNet34_0.01.csv')
# TransferTesting_TargetFGSM(vgg, res34, 0.1, 'transfer_TargetFGSM_VGG19_to_ResNet34_0.1.csv')
# TransferTesting_TargetFGSM(vgg, res34, 0.2, 'transfer_TargetFGSM_VGG19_to_ResNet34_0.2.csv')
#
# TransferTesting_TargetFGSM(vgg, res101, 0.0005, 'transfer_TargetFGSM_VGG19_to_ResNet101_0.0005.csv')
# TransferTesting_TargetFGSM(vgg, res101, 0.0001, 'transfer_TargetFGSM_VGG19_to_ResNet101_0.0001.csv')
# TransferTesting_TargetFGSM(vgg, res101, 0.005, 'transfer_TargetFGSM_VGG19_to_ResNet101_0.005.csv')
# TransferTesting_TargetFGSM(vgg, res101, 0.001, 'transfer_TargetFGSM_VGG19_to_ResNet101_0.001.csv')
# TransferTesting_TargetFGSM(vgg, res101, 0.05, 'transfer_TargetFGSM_VGG19_to_ResNet101_0.05.csv')
# TransferTesting_TargetFGSM(vgg, res101, 0.01, 'transfer_TargetFGSM_VGG19_to_ResNet101_0.01.csv')
# TransferTesting_TargetFGSM(vgg, res101, 0.1, 'transfer_TargetFGSM_VGG19_to_ResNet101_0.1.csv')
# TransferTesting_TargetFGSM(vgg, res101, 0.2, 'transfer_TargetFGSM_VGG19_to_ResNet101_0.2.csv')
#
# TransferTesting_TargetFGSM(vgg, d201, 0.0005, 'transfer_TargetFGSM_VGG19_to_DenseNet201_0.0005.csv')
# TransferTesting_TargetFGSM(vgg, d201, 0.0001, 'transfer_TargetFGSM_VGG19_to_DenseNet201_0.0001.csv')
# TransferTesting_TargetFGSM(vgg, d201, 0.005, 'transfer_TargetFGSM_VGG19_to_DenseNet201_0.005.csv')
# TransferTesting_TargetFGSM(vgg, d201, 0.001, 'transfer_TargetFGSM_VGG19_to_DenseNet201_0.001.csv')
# TransferTesting_TargetFGSM(vgg, d201, 0.05, 'transfer_TargetFGSM_VGG19_to_DenseNet201_0.05.csv')
# TransferTesting_TargetFGSM(vgg, d201, 0.01, 'transfer_TargetFGSM_VGG19_to_DenseNet201_0.01.csv')
# TransferTesting_TargetFGSM(vgg, d201, 0.1, 'transfer_TargetFGSM_VGG19_to_DenseNet201_0.1.csv')
# TransferTesting_TargetFGSM(vgg, d201, 0.2, 'transfer_TargetFGSM_VGG19_to_DenseNet201_0.2.csv')
#
# # DenseNet-201
# TransferTesting_TargetFGSM(d201, anet, 0.0005, 'transfer_TargetFGSM_DenseNet201_to_AlexNet_0.0005.csv')
# TransferTesting_TargetFGSM(d201, anet, 0.0001, 'transfer_TargetFGSM_DenseNet201_to_AlexNet_0.0001.csv')
# TransferTesting_TargetFGSM(d201, anet, 0.005, 'transfer_TargetFGSM_DenseNet201_to_AlexNet_0.005.csv')
# TransferTesting_TargetFGSM(d201, anet, 0.001, 'transfer_TargetFGSM_DenseNet201_to_AlexNet_0.001.csv')
# TransferTesting_TargetFGSM(d201, anet, 0.05, 'transfer_TargetFGSM_DenseNet201_to_AlexNet_0.05.csv')
# TransferTesting_TargetFGSM(d201, anet, 0.01, 'transfer_TargetFGSM_DenseNet201_to_AlexNet_0.01.csv')
# TransferTesting_TargetFGSM(d201, anet, 0.1, 'transfer_TargetFGSM_DenseNet201_to_AlexNet_0.1.csv')
# TransferTesting_TargetFGSM(d201, anet, 0.2, 'transfer_TargetFGSM_DenseNet201_to_AlexNet_0.2.csv')
#
# TransferTesting_TargetFGSM(d201, gnet, 0.0005, 'transfer_TargetFGSM_DenseNet201_to_GoogleNet_0.0005.csv')
# TransferTesting_TargetFGSM(d201, gnet, 0.0001, 'transfer_TargetFGSM_DenseNet201_to_GoogleNet_0.0001.csv')
# TransferTesting_TargetFGSM(d201, gnet, 0.005, 'transfer_TargetFGSM_DenseNet201_to_GoogleNet_0.005.csv')
# TransferTesting_TargetFGSM(d201, gnet, 0.001, 'transfer_TargetFGSM_DenseNet201_to_GoogleNet_0.001.csv')
# TransferTesting_TargetFGSM(d201, gnet, 0.05, 'transfer_TargetFGSM_DenseNet201_to_GoogleNet_0.05.csv')
# TransferTesting_TargetFGSM(d201, gnet, 0.01, 'transfer_TargetFGSM_DenseNet201_to_GoogleNet_0.01.csv')
# TransferTesting_TargetFGSM(d201, gnet, 0.1, 'transfer_TargetFGSM_DenseNet201_to_GoogleNet_0.1.csv')
# TransferTesting_TargetFGSM(d201, gnet, 0.2, 'transfer_TargetFGSM_DenseNet201_to_GoogleNet_0.2.csv')
#
# TransferTesting_TargetFGSM(d201, res34, 0.0005, 'transfer_TargetFGSM_DenseNet201_to_ResNet34_0.0005.csv')
# TransferTesting_TargetFGSM(d201, res34, 0.0001, 'transfer_TargetFGSM_DenseNet201_to_ResNet34_0.0001.csv')
# TransferTesting_TargetFGSM(d201, res34, 0.005, 'transfer_TargetFGSM_DenseNet201_to_ResNet34_0.005.csv')
# TransferTesting_TargetFGSM(d201, res34, 0.001, 'transfer_TargetFGSM_DenseNet201_to_ResNet34_0.001.csv')
# TransferTesting_TargetFGSM(d201, res34, 0.05, 'transfer_TargetFGSM_DenseNet201_to_ResNet34_0.05.csv')
# TransferTesting_TargetFGSM(d201, res34, 0.01, 'transfer_TargetFGSM_DenseNet201_to_ResNet34_0.01.csv')
# TransferTesting_TargetFGSM(d201, res34, 0.1, 'transfer_TargetFGSM_DenseNet201_to_ResNet34_0.1.csv')
# TransferTesting_TargetFGSM(d201, res34, 0.2, 'transfer_TargetFGSM_DenseNet201_to_ResNet34_0.2.csv')
#
# TransferTesting_TargetFGSM(d201, res101, 0.0005, 'transfer_TargetFGSM_DenseNet201_to_ResNet101_0.0005.csv')
# TransferTesting_TargetFGSM(d201, res101, 0.0001, 'transfer_TargetFGSM_DenseNet201_to_ResNet101_0.0001.csv')
# TransferTesting_TargetFGSM(d201, res101, 0.005, 'transfer_TargetFGSM_DenseNet201_to_ResNet101_0.005.csv')
# TransferTesting_TargetFGSM(d201, res101, 0.001, 'transfer_TargetFGSM_DenseNet201_to_ResNet101_0.001.csv')
# TransferTesting_TargetFGSM(d201, res101, 0.05, 'transfer_TargetFGSM_DenseNet201_to_ResNet101_0.05.csv')
# TransferTesting_TargetFGSM(d201, res101, 0.01, 'transfer_TargetFGSM_DenseNet201_to_ResNet101_0.01.csv')
# TransferTesting_TargetFGSM(d201, res101, 0.1, 'transfer_TargetFGSM_DenseNet201_to_ResNet101_0.1.csv')
# TransferTesting_TargetFGSM(d201, res101, 0.2, 'transfer_TargetFGSM_DenseNet201_to_ResNet101_0.2.csv')
#
# TransferTesting_TargetFGSM(d201, vgg, 0.0005, 'transfer_TargetFGSM_DenseNet201_to_VGG19_0.0005.csv')
# TransferTesting_TargetFGSM(d201, vgg, 0.0001, 'transfer_TargetFGSM_DenseNet201_to_VGG19_0.0001.csv')
# TransferTesting_TargetFGSM(d201, vgg, 0.005, 'transfer_TargetFGSM_DenseNet201_to_VGG19_0.005.csv')
# TransferTesting_TargetFGSM(d201, vgg, 0.001, 'transfer_TargetFGSM_DenseNet201_to_VGG19_0.001.csv')
# TransferTesting_TargetFGSM(d201, vgg, 0.05, 'transfer_TargetFGSM_DenseNet201_to_VGG19_0.05.csv')
# TransferTesting_TargetFGSM(d201, vgg, 0.01, 'transfer_TargetFGSM_DenseNet201_to_VGG19_0.01.csv')
# TransferTesting_TargetFGSM(d201, vgg, 0.1, 'transfer_TargetFGSM_DenseNet201_to_VGG19_0.1.csv')
# TransferTesting_TargetFGSM(d201, vgg, 0.2, 'transfer_TargetFGSM_DenseNet201_to_VGG19_0.2.csv')
