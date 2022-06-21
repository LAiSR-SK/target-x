# These are the python libraries that will be used
import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import os
from torch.autograd.gradcheck import zero_gradients


# Target-X, a novel targeted white box attack, that seeks to find the minimum perturbations while maximizing the loss function.
# We define 2 methods "targetx" and "targetx_args" the latter allows the user to pre-define the specific target label for the image
# to be misclassified as in the function header, whereas the "targetx" method asks the user during runtime.

def targetx(image, net, eps=0.05, num_classes=10, overshoot=0.02, max_iter=50):
    """
        :param image: Image of size HxWx3
        :param net: network (input: images, output: values of activation **BEFORE** softmax).
        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
        :param max_iter: maximum number of iterations for targetx (default = 50)
        :param eps: epsilon value for combination of FGSM.
        :return: perturbation from targetx method, number of iterations that it required, new estimated_label, perturbed image, and F_k value (distance moved into new hyperplane)
    """
    # Check if cuda is available.
    is_cuda = torch.cuda.is_available()
    ILSVRClabels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    # If cuda is available use GPU for faster processing, if not, use CPU.
    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    # Convert image into tensor readable by PyTorch, flatten image. f_image is the flattened image
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()

    # Create array of labels I.
    I = (np.array(f_image)).flatten().argsort()[::-1]

    print('I array: ')
    print(I[0:num_classes])

    # target_label is the label number corresponding to the label the image will be misclassified to
    target_label = input('Enter number of label for image to be changed to.')
    new_label = ILSVRClabels[np.int(target_label)].split(',')[0]
    print('Input ID: ' + target_label, ' Label: ' + new_label)

    # Define array as size of specified number of classes
    I = I[0:num_classes]
    # Set first class to the original label.
    orig_label = I[0]

    # print('I array: ')
    # print(I[0:num_classes])

    print('Original Label: ')
    print(I[0])

    # copy image
    input_shape = image.cpu().numpy().shape
    # create pert image variable
    pert_image = copy.deepcopy(image)
    # create leap vector w
    w = np.zeros(input_shape)
    # create r_tot
    r_tot = np.zeros(input_shape)

    # initialize loop variable to 0
    loop_i = 0

    # create a variable x and set it to the original image
    x = Variable(pert_image[None, :], requires_grad=True)
    # forward propogate imgae x through the network
    fs = net.forward(x)
    # list of classes fs_list (possible target labels for input image x)
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    # print fs_list
    print('fs list')
    print(fs_list)
    # setting the initial k, k_i to the original lable orig_label
    k_i = orig_label

    # While label has not changed to custom label and max iterations not reached:
    while k_i != int(target_label) and loop_i < max_iter:

        # Backwards propagate label through graph, get resulting gradient and gradient sign.
        pert = 0  # np.inf we change the to be zero instead of infinty
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        zero_gradients(x)

        # Backwards propagate current label through graph, get resulting gradient and gradient sign.
        # k is how we turn the given target label to an index value
        k = np.where(I == int(target_label))
        print("correct K: ")
        print(k)
        k = k[0][0]
        print('k')
        print(k)

        fs[0, I[k]].backward(retain_graph=True)
        grad_target = x.grad.data.cpu().numpy().copy()
        target_sign_grad = x.grad.sign().cpu().numpy().copy()

        zero_gradients(x)

        # set new w_k, distance between original and current gradient, and new f_k, distance between original and target label
        w = grad_target - grad_orig
        f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

        # Calculate perturbation using targetx formula
        pert = abs(f_k) / np.linalg.norm(w.flatten(), ord=1)
        print('pert')
        print(pert)

        print('----------')

        # compute r_i and r_tot
        r_i = (pert) * w / np.linalg.norm(w)
        # combined calculated perturbation with FGSM, multiply the gradient by the epsilon value
        r_tot = np.float32(r_tot + r_i) + eps * target_sign_grad  # w_k_sign

        # If cuda is being used, store perturbed image tensor in GPU, if not, store it in CPU
        if is_cuda:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
            pert = ((1 + overshoot) * torch.from_numpy(r_tot)) * 100
        else:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)
            pert = ((1 + overshoot) * torch.from_numpy(r_tot))

        # Add new perturbation to x
        x = Variable(pert_image, requires_grad=True)

        # Propagate x through network, get new label, get new f_k distance.
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        print('k_i:')
        print(k_i)
        newf_k = (fs[0, k_i] - fs[0, I[0]]).data.cpu().numpy()
        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, orig_label, k_i, pert_image, pert, newf_k


# Second version of targetx that allows a label to be chosen as function input
def targetx_arg(image, net, target_label, eps=0.05, num_classes=10, overshoot=0.02, max_iter=50):
    """
        :param image: Image of size HxWx3
        :param net: network (input: images, output: values of activation **BEFORE** softmax).
        :param inputNum: Number of label to perturb image to
        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
        :param max_iter: maximum number of iterations for targetx (default = 50)
        :param eps: epsilon value for combination of FGSM.
        :return: perturbation from targetx method, number of iterations that it required, new estimated_label, perturbed image, and F_k value (distance moved into new hyperplane)
    """
    # Check if cuda is available.
    is_cuda = torch.cuda.is_available()
    ILSVRClabels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

    # If cuda is available use GPU for faster processing, if not, use CPU.
    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    # Convert image into tensor readable by PyTorch, flatten image.
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()

    # Create array of labels.
    I = (np.array(f_image)).flatten().argsort()[::-1]

    print('I array: ')
    print(I[0:num_classes])

    newLabel = ILSVRClabels[np.int(target_label)].split(',')[0]
    print('Input ID: ' + str(target_label), ' Label: ' + newLabel)

    # Define array as size of specified number of classes
    I = I[0:num_classes]
    # Set first class to the original label.
    orig_label = I[0]

    # copy image
    input_shape = image.cpu().numpy().shape
    # create pert image variable
    pert_image = copy.deepcopy(image)
    # create leap vector w
    w = np.zeros(input_shape)
    # create r_tot
    r_tot = np.zeros(input_shape)

    newf_k = 0

    # initialize loop variable to 0
    loop_i = 0

    # create a variable x and set it to the original image
    x = Variable(pert_image[None, :], requires_grad=True)
    # forward propogate imgae x through the network
    fs = net.forward(x)
    # setting the initial k, k_i to the original lable orig_label
    k_i = orig_label

    # While label has not changed to custom label and max iterations not reached:
    # While label has not changed to custom label and max iterations not reached:
    while k_i != int(target_label) and loop_i < max_iter:

        # Backwards propagate label through graph, get resulting gradient and gradient sign.
        pert = 0  # np.inf we change the to be zero instead of infinty
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        zero_gradients(x)

        # Backwards propagate current label through graph, get resulting gradient and gradient sign.
        # k is how we turn the given target label to an index value
        k = np.where(I == int(target_label))
        print("correct K: ")
        print(k)
        k = k[0][0]
        print('k')
        print(k)

        fs[0, I[k]].backward(retain_graph=True)
        grad_target = x.grad.data.cpu().numpy().copy()
        target_sign_grad = x.grad.sign().cpu().numpy().copy()

        zero_gradients(x)

        # set new w_k, distance between original and current gradient, and new f_k, distance between original and target label
        w = grad_target - grad_orig
        f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

        # Calculate perturbation using targetx formula
        pert = abs(f_k) / np.linalg.norm(w.flatten(), ord=1)
        print('pert')
        print(pert)

        print('----------')

        # compute r_i and r_tot
        r_i = (pert) * w / np.linalg.norm(w)
        # combined calculated perturbation with FGSM, multiply the gradient by the epsilon value
        r_tot = np.float32(r_tot + r_i) + eps * target_sign_grad  # w_k_sign

        # If cuda is being used, store perturbed image tensor in GPU, if not, store it in CPU
        if is_cuda:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
            pert = ((1 + overshoot) * torch.from_numpy(r_tot)) * 100
        else:
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)
            pert = ((1 + overshoot) * torch.from_numpy(r_tot))

        # Add new perturbation to x
        x = Variable(pert_image, requires_grad=True)

        # Propagate x through network, get new label, get new f_k distance.
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        print('k_i:')
        print(k_i)
        newf_k = (fs[0, k_i] - fs[0, I[0]]).data.cpu().numpy()
        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, orig_label, k_i, pert_image, newf_k

def targetx_return_I_array(image, net, num_classes=10):
    is_cuda = torch.cuda.is_available()
    # If cuda is available use GPU for faster processing, if not, use CPU.
    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    # Convert image into tensor readable by PyTorch, flatten image.
    # f_image is the flattened image
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    # Create array of labels I.
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]

    return I
