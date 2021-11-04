import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets
from PIL import Image
from TargetX import targetx_arg
from models import ResNet, AlexNet
import os
import time
import random
import glob
from imagenet_labels import classes
from torch.autograd import Variable
import torch
import csv


#these are the needed networks that are to be used
#res34 = models.resnet34(pretrained=True)
#res101 = models.resnet101(pretrained=True)
#gnet = models.googlenet(pretrained=True)
#anet = models.alexnet(pretrained=True)
#vgg = models.vgg19(pretrained=True)
#d201 = models.densenet201(pretrained=True)

res34 = ResNet.resnet34()
anet = AlexNet.AlexNet()

stateanet = torch.load("models/alexnet/model.pth")
anet.load_state_dict(stateanet)

stateres34 = torch.load("models/resnet/model.pth")
res34.load_state_dict(stateres34)

#Define transfer testing function, input network to perturb image with, network to test image, epsilon, and csv name.
def TransferTestingHybrid(net, net2, eps, csvname):
    net.cuda()
    net2.cuda()
    net.eval()
    net2.eval()
    Accuracy = 0
    PerturbedAccuracy = 0
    TransferableSuccess = 0
    TargetSuccess = 0
    Net2Correct = 0
    hybridcsv = csvname
    fieldnames = ['Image', 'Correct Label', 'Network 1 Orig Label', 'Network 2 Orig Label', 'Network 1 Pert Label',
              'Network 2 Pert Label']

    # hybrows = []
    counter = 0

    with open(hybridcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(fieldnames)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #Load validation data and label set for ILSVRC
    ILSVRClabels = open(os.path.join('ILSVRC2012validation.txt'), 'r').read().split('\n')
    labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    #Loop through 1000 images of validation dataset, choose random label from 0 to 999, launch attack, test, launch attack on second network, test, write to csv.
    for filename in glob.glob('D:/Imagenet/ILSVRC2012/ILSVRC/Data/CLS-LOC/val/*.JPEG'):  # assuming jpg
        if counter == 1000:
            break
        print(" \n\n\n**************** Hybrid Approach DeepFool 2 *********************\n")
        im_orig = Image.open(filename).convert('RGB')
        print(filename)
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
        while (inputNum == int(correct) or inputNum == detected):
            inputNum = random.randint(0, 999)
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

        def clip_tensor(A, minv, maxv):
            A = torch.max(A, minv * torch.ones(A.shape))
            A = torch.min(A, maxv * torch.ones(A.shape))
            return A

        label2 = net2(im[None, :].cuda())
        label2 = np.argmax(label2.detach().cpu().numpy())
        str_label_orig2 = labels[np.int(label2)].split(',')[0]
        label_pert2 = net2(pert_image)
        label_pert2 = np.argmax(label_pert2.detach().cpu().numpy())
        str_label_pert2 = labels[np.int(label_pert2)].split(',')[0]


        if (int(label_pert2) == int(label_pert)):
            print("Attack was Transferable")
            TransferableSuccess = TransferableSuccess + 1
        if (int(label_pert2) == int(label2)):
            print("Network 2 Perturbed Label = Network 2 Original Label")
            Net2Correct = Net2Correct + 1

        clip = lambda x: clip_tensor(x, 0, 255)
        csvrows = []
        csvrows.append([filename[47:75], str_label_correct, str_label_orig, str_label_orig2, str_label_pert, str_label_pert2])

        with open(csvname, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerows(csvrows)
        counter = counter + 1

    with open(csvname, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(["Epsilon: " + str(eps)])
        csvwriter.writerows(["Accuracy: " + str(Accuracy / 1000)])
        csvwriter.writerows(["Perturbed Accuracy: " + str(PerturbedAccuracy / 1000)])
        csvwriter.writerows(["Perturbed Target Success: " + str(TargetSuccess / 1000)])
        csvwriter.writerows(["Transferability Success: " + str(TransferableSuccess / 1000)])
        csvwriter.writerows(["Net2 Correctness: " + str(Net2Correct / 1000)])

#Define transfer testing function, input network to perturb image with, network to test image, epsilon, and csv name.
def CIFARHybridTransferTesting(original_net, transfer_net, eps, csvname):
    original_net.cuda()
    transfer_net.cuda()
    original_net.eval()
    transfer_net.eval()
    Accuracy = 0
    PerturbedAccuracy = 0
    TransferableSuccess = 0
    TargetSuccess = 0
    Net2Correct = 0
    hybridcsv = csvname
    fieldnames = ['Image', 'Correct Label', 'Network 1 Orig Label', 'Network 2 Orig Label', 'Network 1 Pert Label',
                  'Network 2 Pert Label']
    with open(hybridcsv, 'w', newline='') as csvfile:
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
        print(" \n\n\n**************** Hybrid Approach DeepFool 2 *********************\n")

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

        def clip_tensor(A, minv, maxv):
            A = torch.max(A, minv * torch.ones(A.shape))
            A = torch.min(A, maxv * torch.ones(A.shape))
            return A

        label2 = transfer_net(inputs[None, :].cuda())
        label2 = np.argmax(label2.detach().cpu().numpy())
        str_label_orig2 = classes[np.int(label2)].split(',')[0]
        label_pert2 = transfer_net(pert_image)
        label_pert2 = np.argmax(label_pert2.detach().cpu().numpy())
        str_label_pert2 = classes[np.int(label_pert2)].split(',')[0]

        if (int(label_pert2) == int(label_pert)):
            print("Attack was Transferable")
            TransferableSuccess = TransferableSuccess + 1
        if (int(label_pert2) == int(label2)):
            print("Network 2 Perturbed Label = Network 2 Original Label")
            Net2Correct = Net2Correct + 1

        clip = lambda x: clip_tensor(x, 0, 255)
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
        csvwriter.writerows(["Perturbed Target Success: " + str(TargetSuccess / 1000)])
        csvwriter.writerows(["Transferability Success: " + str(TransferableSuccess / 1000)])
        csvwriter.writerows(["Net2 Correctness: " + str(Net2Correct / 1000)])

# # RESNET34 TESTING
# TransferTestingHybrid(res34, res101, 0.05, 'transfer_hybrid_resnet34_to_resnet101_0.05.csv')
# TransferTestingHybrid(res34, gnet, 0.05, 'transfer_hybrid_resnet34_to_googlenet_0.05.csv')
# TransferTestingHybrid(res34, anet, 0.05, 'transfer_hybrid_resnet34_to_alexnet_0.05.csv')
# TransferTestingHybrid(res34, vgg, 0.05, 'transfer_hybrid_resnet34_to_vggnet_0.05.csv')
# TransferTestingHybrid(res34, d201, 0.05, 'transfer_hybrid_resnet34_to_densenet_0.05.csv')
#
# # RESNET101 TESTING
# TransferTestingHybrid(res101, res34, 0.05, 'transfer_hybrid_resnet101_to_resnet34_0.05.csv')
# TransferTestingHybrid(res101, gnet, 0.05, 'transfer_hybrid_resnet101_to_googlenet_0.05.csv')
# TransferTestingHybrid(res101, anet, 0.05, 'transfer_hybrid_resnet101_to_alexnet_0.05.csv')
# TransferTestingHybrid(res101, vgg, 0.05, 'transfer_hybrid_resnet101_to_vggnet_0.05.csv')
# TransferTestingHybrid(res101, d201, 0.05, 'transfer_hybrid_resnet101_to_densenet_0.05.csv')
#
# # GOOGLENET TESTING
# TransferTestingHybrid(gnet, res34, 0.05, 'transfer_hybrid_googlenet_to_resnet34_0.05.csv')
# TransferTestingHybrid(gnet, res101, 0.05, 'transfer_hybrid_googlenet_to_resnet101_0.05.csv')
# TransferTestingHybrid(gnet, anet, 0.05, 'transfer_hybrid_googlenet_to_alexnet_0.05.csv')
# TransferTestingHybrid(gnet, vgg, 0.05, 'transfer_hybrid_googlenet_to_vggnet_0.05.csv')
# TransferTestingHybrid(gnet, d201, 0.05, 'transfer_hybrid_googlenet_to_densenet_0.05.csv')
#
# # ALEXNET TESTING
# TransferTestingHybrid(anet, res34, 0.05, 'transfer_hybrid_alexnet_to_resnet34_0.05.csv')
# TransferTestingHybrid(anet, res101, 0.05, 'transfer_hybrid_alexnet_to_resnet101_0.05.csv')
# TransferTestingHybrid(anet, gnet, 0.05, 'transfer_hybrid_alexnet_to_googlenet_0.05.csv')
# TransferTestingHybrid(anet, vgg, 0.05, 'transfer_hybrid_alexnet_to_vggnet_0.05.csv')
# TransferTestingHybrid(anet, d201, 0.05, 'transfer_hybrid_alexnet_to_densenet_0.05.csv')
#
# # VGG TESTING
# TransferTestingHybrid(vgg, res34, 0.05, 'transfer_hybrid_vggnet_to_resnet34_0.05.csv')
# TransferTestingHybrid(vgg, res101, 0.05, 'transfer_hybrid_vggnet_to_resnet101_0.05.csv')
# TransferTestingHybrid(vgg, gnet, 0.05, 'transfer_hybrid_vggnet_to_googlenet_0.05.csv')
# TransferTestingHybrid(vgg, anet, 0.05, 'transfer_hybrid_vggnet_to_alexnet_0.05.csv')
# TransferTestingHybrid(vgg, d201, 0.05, 'transfer_hybrid_vggnet_to_densenet_0.05.csv')
#
# # D201 TESTING
# TransferTestingHybrid(d201, res34, 0.05, 'transfer_hybrid_densenet_to_resnet34_0.05.csv')
# TransferTestingHybrid(d201, res101, 0.05, 'transfer_hybrid_densenet_to_resnet101_0.05.csv')
# TransferTestingHybrid(d201, gnet, 0.05, 'transfer_hybrid_densenet_to_googlenet_0.05.csv')
# TransferTestingHybrid(d201, anet, 0.05, 'transfer_hybrid_densenet_to_alexnet_0.05.csv')
# TransferTestingHybrid(d201, vgg, 0.05, 'transfer_hybrid_densenet_to_vggnet_0.05.csv')

#CIFAR10 TESTING
CIFARHybridTransferTesting(res34, anet, 0.05, 'transfer_hybrid_resnet34_to_alexnet_0.05_CIFAR10.csv')
CIFARHybridTransferTesting(anet, res34, 0.05, 'transfer_hybrid_alexnet_to_resnet34_0.05_CIFAR10.csv')