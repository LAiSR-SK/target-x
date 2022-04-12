import random

import art
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from TargetX import targetx_return_I_array

# im_orig = Image.open('pictures/buffalo.JPEG')
im_orig = Image.open('pictures/cat.JPEG')
# im_orig = Image.open('pictures/crocodile.JPEG')
# im_orig = Image.open('pictures/dog.JPEG')
# im_orig = Image.open('pictures/dog2.JPEG')
# im_orig = Image.open('pictures/elephant.JPEG')
# im_orig = Image.open('pictures/frog.JPEG')
# im_orig = Image.open('pictures/llama.JPEG')
# im_orig = Image.open('pictures/orangutan.JPEG')
# im_orig = Image.open('pictures/owl.JPEG')
# im_orig = Image.open('pictures/seal.jpg')



net = models.resnet34(pretrained=True)
net.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


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

tf_mod = transforms.Compose([transforms.ToPILImage(),
                             transforms.CenterCrop(244)])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])(im_orig)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

classifier = art.estimators.classification.PyTorchClassifier(
    model=net,
    input_shape=(3, 224, 224),
    loss=criterion,
    optimizer=optimizer,
    nb_classes=1000
)

I = targetx_return_I_array(preprocess, net, 10)
print(I)

# label to exclude from choice
exclude_label = I[0]
# chooses a random int in the I array
inputNum = random.choice(I)
if inputNum == exclude_label:
    print("target label same as original label")
    inputNum = random.choice(I)

print('Generated Eval Label: ', inputNum)
targetLabel = np.array([])
targetLabel = np.append(targetLabel, inputNum)

input_tensor = preprocess
input_batch = input_tensor.unsqueeze(0)
print(input_batch.shape)

a = classifier.predict(input_batch, 1, False)

label_orig = np.argmax(a.flatten())
print(label_orig)

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[np.int(label_orig)].split(',')[0]
print("Original Label: ", str_label_orig)

# Targeted Testing
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.0001, norm=np.inf, targeted=True)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.0005, norm=np.inf, targeted=True)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.001, norm=np.inf, targeted=True)
targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.005, norm=np.inf, targeted=True)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.01, norm=np.inf, targeted=True)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.05, norm=np.inf, targeted=True)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.1, norm=np.inf, targeted=True)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.2, norm=np.inf, targeted=True)

# Un-Targeted Testing
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.0001, norm=np.inf, targeted=False)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.0005, norm=np.inf, targeted=False)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.001, norm=np.inf, targeted=False)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.005, norm=np.inf, targeted=False)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.01, norm=np.inf, targeted=False)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.05, norm=np.inf, targeted=False)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.1, norm=np.inf, targeted=False)
# targetFGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.2, norm=np.inf, targeted=False)

input_array = input_batch.numpy()
# targeted
adv_im = targetFGSM.generate(x=input_array, y=targetLabel)
# un-targeted
# adv_im = targetFGSM.generate(x=input_array)

b = classifier.predict(adv_im, 1, False)
label_pert = np.argmax(b.flatten())
str_label_pert = labels[np.int(label_pert)].split(',')[0]
print("Perturbed Label: ", str_label_pert)

orig_im = input_array.squeeze()
pert_im = adv_im.squeeze()
orig_im = orig_im.swapaxes(0, 1)  # (3,224,224) -> (224,3,224)
orig_im = orig_im.swapaxes(1, 2)  # (224,3,224) -> (224,224,3)

print(orig_im.shape)
print(input_tensor.shape)
print(orig_im)
print(input_tensor)

plt.figure()
plt.imshow(tf(input_tensor))
plt.title(str_label_orig)
plt.show()

plt.imshow(tf(torch.from_numpy(pert_im)))
plt.title(str_label_pert)
plt.show()

targetFGSM_pert = torch.from_numpy(pert_im) - input_tensor
plt.imshow(tf_mod(targetFGSM_pert))
plt.title(str_label_pert)
plt.show()
