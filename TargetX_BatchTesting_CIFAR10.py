# Basic testing for evaluating Target-X on CIFAR-10

# These are the python libraries that will be used
import torchvision.transforms as transforms
import numpy as np
import torch
import random
from models import AlexNet, ResNet, GoogleNet
from TargetX import targetx_arg
import time
import csv
import torchvision.datasets as datasets

# these are the needed networks that are to be used
res34 = ResNet.resnet34()
anet = AlexNet.AlexNet()
gnet = GoogleNet.GoogLeNet()

# Define networks for testing, load pretrained model on CIFAR10
stateanet = torch.load("models/alexnet/model.pth")
anet.load_state_dict(stateanet)

stateres34 = torch.load("models/resnet/model_res34.pth")
res34.load_state_dict(stateres34)

stategnet = torch.load("models/googlenet/model.pth")
gnet.load_state_dict(stategnet)

# Define testing function, network, epsilon, and name of csv
def CIFARTestingFunction(net, eps, targetcsv):
    net.eval()
    net.cuda()

    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]

    fieldnames = ['Image', 'Original Label', 'Classified Label Before Perturbation', 'Targeted Label', 'Perturbed Label', 'Execution Time', 'F_k', 'Avg Difference', 'Frobenius of Difference']

    with open(targetcsv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fieldnames)

    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

    targetx_csv = targetcsv
    with open(targetx_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fieldnames)

    # Define classes for CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Define accuracy and success measures and counter variable.
    Accuracy = 0
    PerturbedAccuracy = 0
    TargetSuccess = 0
    PerturbedAvgFk = 0
    PerturbedAvgDiff = 0
    PerturbedAvgFroDiff = 0
    counter = 0

    # Loop through CIFAR testset for 1000 iterations, select random label from 0 to 9 to perturb image to, run attack, test metrics.
    for i, data in enumerate(testset): # assuming jpg
        inputs, labels = data
        print(i)
        if counter == 1000:
            break

        # Selects random class number
        inputNum = random.randint(0, 9)
        # Prints the generated label
        print('Generated Eval Label: ', inputNum)
        # Checks to see if the evaluation label is the same as the correct label,
        # Generates a new label if it is.
        correct = labels
        detect = net.forward(inputs[None, ...].cuda())
        detected = np.argmax(detect.data.cpu().numpy().flatten())
        str_label_detect = classes[np.int(detected)].split(',')[0]
        print('Detected: ', str_label_detect)
        while(inputNum == int(correct) or inputNum == detected):
            inputNum = random.randint(0, 9)
            print('Eval same as correct/detected. Generating new input: ', inputNum)
        str_label_correct = classes[np.int(correct)].split(',')[0]
        str_label_target = classes[np.int(inputNum)].split(',')[0]
        print('Target: ', str_label_target)
        print('Correct: ', str_label_correct)

        # The evaluation between the original and perturbed class labels
        start_time = time.time()
        r, loop_i, label_orig, label_pert, pert_image, pert, newf_k = targetx_arg(inputs, net, inputNum, eps)
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

        # Clips the tensor value
        def clip_tensor(A, minv, maxv):
            A = torch.max(A, minv * torch.ones(A.shape))
            A = torch.min(A, maxv * torch.ones(A.shape))
            return A

        # calling the clipping function for "x"
        clip = lambda x: clip_tensor(x, 0, 255)

        # Normalizes the image
        imagetransform = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                                             transforms.Normalize(list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                                             transforms.Lambda(clip)])

        diff = imagetransform(pert_image.cpu()[0]) - i
        # Calculate frobenius of difference
        fro = np.linalg.norm(diff.numpy())
        # Calculate average distance
        average = torch.mean(torch.abs(diff))
        csvrows = []
        PerturbedAvgFk = PerturbedAvgFk + newf_k
        PerturbedAvgDiff = PerturbedAvgDiff + average
        PerturbedAvgFroDiff = PerturbedAvgFroDiff + fro
        # Append values to rows, append to csv file
        csvrows.append([i, str_label_correct, str_label_orig, str_label_target, str_label_pert, torch.cuda.memory_stats('cuda:0')['active.all.current'], str(loop_i), str(execution_time), newf_k, average, fro])
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
