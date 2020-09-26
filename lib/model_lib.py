# This is used for initialize model
# modified according to PyTorch Official Tutorial
# By Ruibing 2020/09/08 09:53 a.m.

import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True, aux_logits = False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if use_pretrained:
        print("=> using pre-trained model '{}'".format(model_name))
    else:
        print("=> creating model '{}'".format(model_name))

    if model_name.startswith("resnet"):
        """ Resnet
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name.startswith("alexnet"):
        """ Alexnet
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name.startswith("vgg"):
        """ VGG
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name.startswith("squeezenet"):
        """ Squeezenet
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name.startswith("densenet"):
        """ Densenet
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name.startswith("inception"):
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
    elif model_name.startswith("googlenet"):
        model_ft = models.__dict__[model_name](pretrained=use_pretrained, aux_logits = aux_logits)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        if aux_logits:
            num_ftrs = model_ft.aux1.in_features
            model_ft.aux1 = nn.Linear(num_ftrs, num_classes)
            num_ftrs = model_ft.aux2.in_features
            model_ft.aux2 = nn.Linear(num_ftrs, num_classes)

        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size