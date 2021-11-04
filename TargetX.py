import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import os
from torch.autograd.gradcheck import zero_gradients

#Combined Hybrid algorithm of Deepfool and FGSM, plus modification to obtain maximum hyperplanes.

def targetx(image, net, eps=0.05, num_classes=1000, overshoot=0.02, max_iter=50):
    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :param eps: epsilon value for combination of FGSM.
       :return: perturbation from hybrid method, number of iterations that it required, new estimated_label, perturbed image, and F_k value (distance moved into new hyperplane)
    """
    #Check if cuda is available.
    is_cuda = torch.cuda.is_available()
    ILSVRClabels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    #If cuda is available use GPU for faster processing, if not, use CPU.
    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    #Convert image into tensor readable by PyTorch, flatten image.
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    #Create array of labels.
    I = (np.array(f_image)).flatten().argsort()[::-1]
    #I = np.sort(I)
    print(I)
    inputNum = input('Enter number of label for image to be changed to.')
    newLabel = ILSVRClabels[np.int(inputNum)].split(',')[0]
    print('Input ID: ' + inputNum, ' Label: ' + newLabel)

    #Define array as size of specified number of classes, set first class to the original label.
    I = I[0:num_classes]
    label = I[0]
    print('I array:')
    print(I[0:num_classes])
    print(I[0])


    #Copy the image, create variable for perturbed image, as well as w and r_tot, using the shape of the image
    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    #initialize loop variable to 0
    loop_i = 0

    #Set x to the original image, forward propagate it through the network, get list of classes
    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    print('fs list')
    print(fs_list)
    k_i = label

    #While label has not changed to custom label and max iterations not reached:
    while k_i != int(inputNum) and loop_i < max_iter:

        #Backwards propagate label through graph, get resulting gradient and gradient sign.
        pert = 0  # np.inf we change the to be zero instead of infinty
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()
        grad_orig_sign = x.grad.sign().cpu().numpy().copy()  # added for fgsm

        zero_gradients(x)
             #Backwards propagate current label through graph, get resulting gradient and gradient sign.
        k = np.where(I == int(inputNum))
        k = k[0][0]
        print('k')
        print(k)
        fs[0, I[k]].backward(retain_graph=True)
        print(I[k])
        print('0, I[k]')
        print(fs[0, I[k]])
        cur_grad = x.grad.data.cpu().numpy().copy()
        cur_sign_grad = x.grad.sign().cpu().numpy().copy()  # added for fgsm

        # set new w_k and new f_k
        w_k = cur_grad - grad_orig
        w_k_sign = cur_sign_grad - grad_orig_sign  # added for fgsm
        f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()


        #Calculate perturbation using deepfool formula
        pert_k = abs(f_k) / np.linalg.norm(w_k.flatten(), ord=1)
        print('pert_k')
        print(pert_k)

        print('----------')

        # determine which w_k to use
        if pert_k > pert:  # we change here the "<" to be ">" to get the max hyperplanes
            pert = pert_k
            w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        #combined calculated perturbation with FGSM, multiply the gradient by the epsilon value
        r_tot = np.float32(r_tot + r_i) + eps * cur_sign_grad #w_k_sign

        #If cuda is being used, store perturbed image tensor in GPU, if not, store it in CPU
        if is_cuda:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
            pert = ((1 + overshoot) * torch.from_numpy(r_tot))*100
        else:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)
            pert = ((1 + overshoot) * torch.from_numpy(r_tot))

        #Add new perturbation to x
        x = Variable(pert_image, requires_grad=True)

        #Propagate x through network, get new label, get new f_k distance.
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        print('k_i:')
        print(k_i)
        newf_k = (fs[0, k_i] - fs[0, I[0]]).data.cpu().numpy()
        loop_i += 1


    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image, pert, newf_k

#Second version of targetx that allows a label to be chosen as function input
def targetx_arg(image, net, inputNum, eps=0.05, num_classes=1000, overshoot=0.02, max_iter=50):
    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param inputNum: Number of label to perturb image to
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :param eps: epsilon value for combination of FGSM.
       :return: perturbation from hybrid method, number of iterations that it required, new estimated_label, perturbed image, and F_k value (distance moved into new hyperplane)
    """
    #Check if cuda is available.
    is_cuda = torch.cuda.is_available()
    ILSVRClabels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    #If cuda is available use GPU for faster processing, if not, use CPU.
    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    #Convert image into tensor readable by PyTorch, flatten image.
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    #Create array of labels.
    I = (np.array(f_image)).flatten().argsort()[::-1]
    #I = np.sort(I)
    newLabel = ILSVRClabels[np.int(inputNum)].split(',')[0]
    print('Input ID: ' + str(inputNum), ' Label: ' + newLabel)

    #Define array as size of specified number of classes, set first class to the original label.
    I = I[0:num_classes]
    label = I[0]


    #Copy the image, create variable for perturbed image, as well as w and r_tot, using the shape of the image
    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    #initialize loop variable to 0
    loop_i = 0

    #Set x to the original image, forward propagate it through the network, get list of classes
    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    #While label has not changed to custom label and max iterations not reached:
    while k_i != int(inputNum) and loop_i < max_iter:

        #Backwards propagate label through graph, get resulting gradient and gradient sign.
        pert = 0  # np.inf we change the to be zero instead of infinty
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        zero_gradients(x)
             #Backwards propagate current label through graph, get resulting gradient and gradient sign.
        k = np.where(I == int(inputNum))
        k = k[0][0]
        fs[0, I[k]].backward(retain_graph=True)
        cur_grad = x.grad.data.cpu().numpy().copy()
        cur_sign_grad = x.grad.sign().cpu().numpy().copy()  # added for fgsm

        # set new w_k and new f_k
        w_k = cur_grad - grad_orig
        f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()


        #Calculate perturbation using deepfool formula
        pert_k = abs(f_k) / np.linalg.norm(w_k.flatten(), ord=1)


        # determine which w_k to use
        if pert_k > pert:  # we change here the "<" to be ">" to get the max hyperplanes
            pert = pert_k
            w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        #combined calculated perturbation with FGSM, multiply the gradient by the epsilon value
        r_tot = np.float32(r_tot + r_i) + eps * cur_sign_grad #w_k_sign

        #If cuda is being used, store perturbed image tensor in GPU, if not, store it in CPU
        if is_cuda:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)
            pert = ((1 + overshoot) * torch.from_numpy(r_tot))

        #Add new perturbation to x
        x = Variable(pert_image, requires_grad=True)

        #Propagate x through network, get new label, get new f_k distance.
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        newf_k = (fs[0, k_i] - fs[0, I[0]]).data.cpu().numpy()
        loop_i += 1


    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image, newf_k