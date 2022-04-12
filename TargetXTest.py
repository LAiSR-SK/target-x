import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from PIL import Image
from TargetX import targetx, targetx_return_I_array, targetx_arg
import os
import time

#Define test network
net = models.resnet34(pretrained=True)

# Switch to evaluation mode
net.eval()

#Open image
# im_orig = Image.open('new/ILSVRC2017_test_00004355.JPEG')
# im_orig = Image.open('pictures/buffalo.JPEG')
# im_orig = Image.open('pictures/cat.JPEG')
im_orig = Image.open('pictures/owl.JPEG')


mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]


# Remove the mean
im = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])(im_orig)

#Run attack
I = targetx_return_I_array(im, net, 10)
print(I)
start_time = time.time()
r, loop_i, label_orig, label_pert, pert_image, pert, newf_k = targetx(im, net, 0.001)
# r, loop_i, label_orig, label_pert, pert_image, pert, newf_k = targetx_arg(im, net, 318, 0.001)
end_time = time.time()
exec_time  = end_time-start_time
print(exec_time)
#Open labels
labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

#Get original and perturbed labels
str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

# str_label_orig = fs_list[np.int(label_orig)].split(',')[0]
# str_label_pert = fs_list[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

#clip tensor
def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)


tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                        transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224)])

# plt.figure()
# plt.imshow(tf(im.cpu()))
# plt.title(str_label_orig)
# plt.show()
#
# plt.figure()
# plt.imshow(tf(pert_image.cpu()[0]))
# plt.title(str_label_pert)
# plt.show()
#
# plt.figure()
# plt.imshow(tf(pert.cpu()[0]))
# plt.title(str_label_pert)
# plt.show()

plt.figure()
plt.imshow(tf(im.cpu()))
plt.title(str_label_orig)
tf(im.cpu()).save('orig.png')
plt.show()
plt.figure()
plt.imshow(tf(pert_image.cpu()[0]))
img = tf(pert_image.cpu()[0])
img.save('image.png')
plt.title(str_label_pert)
plt.show()
print(loop_i)
plt.figure()
fc = 100000
plt.imshow(tf(pert.cpu()[0]*fc))
tf(pert.cpu()[0]*fc).save('perturbation.png')
plt.title(str_label_pert)
plt.show()
