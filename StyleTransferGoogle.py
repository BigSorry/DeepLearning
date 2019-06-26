from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import glob
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import Losses
import VGG as net

import os
import time
import pickle
from itertools import combinations


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

# desired size of the output image
imsize = 56 if torch.cuda.is_available() else 56# use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)  # pause a bit so that plots are updated

def image_loader(image_name, resize):
    image = Image.open(image_name)
    if resize == True:
        image = image.resize((imsize, imsize), Image.ANTIALIAS)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)

    return image.to(device, torch.float)

def getConfigsStyle(model, style_length):
    convolutions = []
    i = 0
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
            convolutions.append(name)

    options = list(combinations(convolutions, style_length))
    return  options

image_list = []
image_names = []
imagePerMap = 1
for map in glob.glob('art/*'):
    count = 0
    for image in glob.glob("{}/*.jpg".format(map)):
        image_names.append(os.path.basename(image))
        im = image_loader(image, True)
        image_list.append(im)
        count+=1
        if count == imagePerMap:
            break

content_img = image_loader("images/dancing.jpg", True)
# image_names = [image_names[i] for i in indices]
# image_list = [image_list[i] for i in indices]
models = {'vgg11' : models.vgg11_bn(pretrained=True).features.to(device).eval(), 'vgg13': models.vgg13_bn(pretrained=True).features.to(device).eval(),
           'vgg16' : models.vgg16_bn(pretrained=True).features.to(device).eval(), 'vgg19' : models.vgg19_bn(pretrained=True).features.to(device).eval()}
plotInfo = {name: [] for name in models.keys()}
# for name in models.keys():
#     try:
#         os.makedirs("images/output/{}".format(name))
#     except FileExistsError:
#         pass
imgnumber = 0
for style_img in image_list:
    assert style_img.size() == content_img.size(), "we need to import style and content images of the same size"
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, len(models)+2, 2)
    ax.title.set_text('Style Image')
    ax.imshow(style_img.cpu()[0].permute(1, 2, 0))
    ax2 = fig.add_subplot(1, len(models)+2, 1, sharex=ax, sharey=ax)
    ax2.title.set_text('Content Image')
    ax2.imshow(content_img.cpu()[0].permute(1, 2, 0))
    indexSubPlot = 3
    for modelName, cnn in models.items():
        start = time.time()
        content_layers_default = ['conv_4']
        style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        class Normalization(nn.Module):
            def __init__(self, mean, std):
                super(Normalization, self).__init__()
                # .view the mean and std to make them [C x 1 x 1] so that they can
                # directly work with image Tensor of shape [B x C x H x W].
                # B is batch size. C is number of channels. H is height and W is width.
                self.mean = mean.clone().detach().view(-1, 1, 1)
                self.std = std.clone().detach().view(-1, 1, 1)

            def forward(self, img):
                # normalize img
                return (img - self.mean) / self.std



        def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                       style_img, content_img,
                                       content_layers=content_layers_default,
                                       style_layers=style_layers_default):
            cnn = copy.deepcopy(cnn)

            # normalization module
            normalization = Normalization(normalization_mean, normalization_std).to(device)

            # just in order to have an iterable access to or list of content/syle
            # losses
            content_losses = []
            style_losses = []

            # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
            # to put in modules that are supposed to be activated sequentially
            #model = nn.Sequential(normalization)
            model = nn.Sequential()
            i = 0  # increment every time we see a conv
            convolutionPart = cnn.children()
            #convolutionPart = list(cnn.children())[0]
            for layer in convolutionPart:
                if isinstance(layer, nn.Conv2d):
                    i += 1
                    name = 'conv_{}'.format(i)
                elif isinstance(layer, nn.ReLU):
                    name = 'relu_{}'.format(i)
                    # The in-place version doesn't play very nicely with the ContentLoss
                    # and StyleLoss we insert below. So we replace with out-of-place
                    # ones here.
                    layer = nn.ReLU(inplace=False)
                elif isinstance(layer, nn.MaxPool2d):
                    name = 'pool_{}'.format(i)
                elif isinstance(layer, nn.BatchNorm2d):
                    name = 'bn_{}'.format(i)
                elif isinstance(layer, nn.Sequential):
                    name = 'seq_{}'.format(i)
                elif isinstance(layer, nn.AdaptiveAvgPool2d):
                    name = 'avgpool_{}'.format(i)
                elif isinstance(layer, nn.Linear):
                    name = 'linear_{}'.format(i)
                else:
                    raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

                model.add_module(name, layer)

                if name in content_layers:
                    # add content loss:
                    target = model(content_img).detach()
                    content_loss = Losses.ContentLoss(target)
                    model.add_module("content_loss_{}".format(i), content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    target_feature = model(style_img).detach()
                    style_loss = Losses.StyleLoss(target_feature)
                    model.add_module("style_loss_{}".format(i), style_loss)
                    style_losses.append(style_loss)

            # now we trim off the layers after the last content and style losses
            for i in range(len(model) - 1, -1, -1):
                if isinstance(model[i], Losses.ContentLoss) or isinstance(model[i], Losses.StyleLoss):
                    break

            model = model[:(i + 1)]

            return model, style_losses, content_losses

        input_img = content_img.clone()
        # if you want to use white noise instead uncomment the below line:
        # input_img = torch.randn(content_img.data.size(), device=device)

        # add the original input image to the figure:
        def get_input_optimizer(input_img):
            # this line to show that input is a parameter that requires a gradient
            optimizer = optim.LBFGS([input_img.requires_grad_()])
            return optimizer

        def run_style_transfer(cnn, normalization_mean, normalization_std,
                               content_img, style_img, input_img, num_steps=300,
                               style_weight=1000000, content_weight=1):
            """Run the style transfer."""
            print('Building the style transfer model..')
            model, style_losses, content_losses = get_style_model_and_losses(cnn,
                normalization_mean, normalization_std, style_img, content_img)
            optimizer = get_input_optimizer(input_img)

            print('Optimizing..')
            run = [0]
            totalLoss = 2**32-1
            while totalLoss >= 0.2 and run[0] <= num_steps:

                def closure():
                    # correct the values of updated input image
                    input_img.data.clamp_(0, 1)

                    optimizer.zero_grad()
                    model(input_img)
                    style_score = 0
                    content_score = 0

                    for sl in style_losses:
                        style_score += sl.loss
                    for cl in content_losses:
                        content_score += cl.loss

                    style_score *= style_weight
                    content_score *= content_weight

                    loss = style_score + content_score
                    loss.backward()

                    run[0] += 1
                    if run[0] % 50 == 0:
                        print("run {}:".format(run))
                        print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                            style_score, content_score))
                        print()

                    return style_score + content_score

                totalLoss = optimizer.step(closure)

            # a last correction...
            input_img.data.clamp_(0, 1)
            print("run {}:".format(run))
            return input_img, run[0], totalLoss
        output, runs, loss = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img)
									


        ax3 = fig.add_subplot(1, len(models)+2, indexSubPlot, sharex=ax, sharey=ax)
        ax3.title.set_text('Output Image with architecture {}'.format(modelName))
        ax3.imshow(output.cpu()[0].permute(1, 2, 0).detach().numpy())
        indexSubPlot+=1

        result = transforms.ToPILImage()(output.cpu()[0])
        elapsedSeconds = int(time.time() - start) / 10e9
        plotInfo[modelName].append((image_names[imgnumber], loss.item(), elapsedSeconds, runs))

    fig.savefig('images/output/art{}'.format(image_names[imgnumber]))
    imgnumber += 1

infoName = "info_dict.pickle"
with open(infoName, 'wb') as file:
	pickle.dump(plotInfo, file, protocol = pickle.HIGHEST_PROTOCOL)
	
