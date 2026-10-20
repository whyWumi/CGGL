import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
import inversefed
import torchvision.utils as vutils
from torchvision import models, datasets, transforms
from collections import defaultdict, OrderedDict
from copy import deepcopy
import re
import copy
import time
import math
import logging

from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.utils import download_and_extract_archive
import numpy as np
import sys
import os
from PIL import Image


import global_var

device = global_var.get_device()
args = inversefed.options().parse_args()
setup = inversefed.utils.system_startup(args)
start_time = time.time()


def MyImageNet_model_generator(model_name="Resnet18"):
    from torchvision.models import resnet18

    model = resnet18(pretrained=True)
    model.fc = nn.Sequential(
        # nn.Linear(2048,512,bias=True),
        nn.Linear(512, 256, bias=True)
    )
    return model.to(device)

    # model = torchvision.models.resnet50(pretrained=True)
    # model.fc = nn.Sequential(
    #     # nn.Linear(2048,512,bias=True),
    #     nn.Linear(512, 200, bias=True)
    # )

    # if args.pretrained is not None:
    #     if os.path.isfile(args.pretrained):
    #         if 'moco' not in args.pretrained:
    #             if args.pretrained.endswith('tar'):
    #                 checkpoint = torch.load(
    #                     args.pretrained, map_location='cpu')
    #                 state_dict = checkpoint['state_dict']
    #                 for k in list(state_dict.keys()):
    #                     if k.startswith('module.'):
    #                         state_dict[k[len("module."):]] = state_dict[k]
    #                     del state_dict[k]
    #                 model.load_state_dict(state_dict)
    #             elif args.pretrained.endswith('pth'):
    #                 model.load_state_dict(torch.load(args.pretrained))
    #             else:
    #                 raise ValueError('args.pretrained file naming format \
    #                                     is incorrect, should be \
    #                                     .pth (parameters) or .tar (checkpoint)')
    #         else:
    #             print("loading MoCoV2 checkpoint'{}'".format(args.pretrained))
    #             checkpoint = torch.load(args.pretrained, map_location="cpu")
    #             # rename moco pre-trained keys
    #             state_dict = checkpoint['state_dict']
    #             for k in list(state_dict.keys()):
    #                 # retain only encoder_q up to before the embedding layer
    #                 if (k.startswith('module.encoder_q')
    #                         and not k.startswith('module.encoder_q.fc')):
    #                     # remove prefix
    #                     state_dict[k[len("module.encoder_q."):]] = \
    #                         state_dict[k]
    #                 # delete renamed or unused k
    #                 del state_dict[k]
    #
    #             msg = model.load_state_dict(state_dict, strict=False)
    #             assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    #
    #         print("=> loaded pre-trained model '{}'".format(args.pretrained))
    #     else:
    #         print("=> no model found at '{}'".format(args.pretrained))

    # model.to(**setup)
    # model.eval()
    return model.to(device)


def TinyImageNet_model_generator(model_name="Resnet18"):
    if model_name == "Resnet18":
        from torchvision.models import resnet18

        model = resnet18(pretrained=True)
        model.fc = nn.Sequential(
            # nn.Linear(2048,512,bias=True),
            nn.Linear(512, 200, bias=True)
        )
        return model.to(device)
    else:
        from torchvision.models import vgg16_bn

        model = vgg16_bn(pretrained=True)
        model.classifier.append(nn.ReLU(inplace=True))
        model.classifier.append(nn.Dropout(p=0.5, inplace=False))
        model.classifier.append(nn.Linear(1000, 200))
        return model.to(device)


def Cifar100_model_generator(model_name="Resnet18"):
    if model_name == "Resnet18":
        from torchvision.models import resnet18

        model = resnet18(pretrained=True)
        # model = resnet18(weights='IMAGENET1K_V1')
        model.conv1 = nn.Conv2d(
            model.conv1.in_channels, model.conv1.out_channels, 3, 1, 1
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Sequential(
            # nn.Linear(2048,512,bias=True),
            nn.Linear(512, 100, bias=True)
        )
        return model.to(device)
    else:
        from torchvision.models import vgg16_bn

        model = vgg16_bn(pretrained=True)
        model.classifier.append(nn.ReLU(inplace=True))
        model.classifier.append(nn.Dropout(p=0.5, inplace=False))
        model.classifier.append(nn.Linear(1000, 100))
        return model.to(device)


def CalTech256_model_generator(model_name="Resnet18"):
    if model_name == "Resnet18":
        from torchvision.models import resnet18

        model = resnet18(pretrained=True)
        model.fc = nn.Sequential(
            # nn.Linear(2048,512,bias=True),
            nn.Linear(512, 256, bias=True)
        )
        return model.to(device)
    else:
        from torchvision.models import vgg16_bn

        model = vgg16_bn(pretrained=True)
        model.classifier.append(nn.ReLU(inplace=True))
        model.classifier.append(nn.Dropout(p=0.5, inplace=False))
        model.classifier.append(nn.Linear(1000, 256))
        return model.to(device)
