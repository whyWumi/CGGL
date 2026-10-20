from cggl_paths import prepare_directories

prepare_directories()

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

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(8, sci_mode=True)

global_var.set_device(device)
torch.cuda.is_available()
from dataset import *
from common_DL import *
from gradient_lib import *
from federated_learning import *
from model_structure import *
from utils import *
from inversion_attacks import *
from model_structure import *

from Main_framework_lib2 import *

# Dataset
Cifar100_train_dataset = get_dataset(
    "Cifar100", train=True, transform=None, download=True
)
Cifar100_test_dataset = get_dataset(
    "Cifar100", train=False, transform=None, download=True
)

Cifar100_train_loader = torch.utils.data.DataLoader(
    Cifar100_train_dataset, batch_size=128, shuffle=True
)
Cifar100_test_loader = torch.utils.data.DataLoader(
    Cifar100_test_dataset, batch_size=128, shuffle=False
)
# TinyImageNet_train_dataset = get_dataset("TinyImageNet",train=True,transform=None,download=True)
# TinyImageNet_test_dataset= get_dataset("TinyImageNet",train=False,transform=None,download=True)
#
# TinyImageNet_train_loader = torch.utils.data.DataLoader(TinyImageNet_train_dataset,batch_size=32,shuffle=True)
# TinyImageNet_test_loader = torch.utils.data.DataLoader(TinyImageNet_test_dataset,batch_size=32,shuffle=False)
# CalTech256_train_dataset = get_dataset("CalTech256",train=True,transform=None,download=True)
# CalTech256_test_dataset= get_dataset("CalTech256",train=False,transform=None,download=True)

# CalTech256_train_loader = torch.utils.data.DataLoader(CalTech256_train_dataset,batch_size=16,shuffle=True)
# CalTech256_test_loader = torch.utils.data.DataLoader(CalTech256_test_dataset,batch_size=16,shuffle=False)
test_dataset_dict = {
    # "TinyImageNet":TinyImageNet_test_dataset,
    "Cifar100": Cifar100_test_dataset,
    # "CalTech256":CalTech256_test_dataset
}

train_dataset_dict = {
    # "TinyImageNet":TinyImageNet_train_dataset,
    "Cifar100": Cifar100_train_dataset,
    # "CalTech256":CalTech256_train_dataset
}
test_loader_dict = {
    # "TinyImageNet":TinyImageNet_test_loader,
    "Cifar100": Cifar100_test_loader,
    # "CalTech256":CalTech256_test_loader
}

train_loader_dict = {
    # "TinyImageNet":TinyImageNet_train_loader,
    "Cifar100": Cifar100_train_loader,
    # "CalTech256":CalTech256_train_loader
}
data_info = {
    "test_dataset_dict": test_dataset_dict,
    "train_dataset_dict": train_dataset_dict,
    "test_loader_dict": test_loader_dict,
    "train_loader_dict": train_loader_dict,
}
# Exp
# Cifar100
# Config
dataset_name = "Cifar100"
label_range = 100
each_class_num = 400
dataset_size = 40000

batch_size = 32
client_num = 50
malicious_num = 10
target_class = 0
knowledge_level = "FK"  # FK,SK,NK
agr_name = "MKrum"  # MKrum,Bulyan,AFA,Fang,Standard
model_save_path = "./model/"
normal_model_name = "{}_Normal_Model.pth".format(dataset_name)
malicious_model_name = "{}_{}_{:d}_Malicious_Model.pth".format(
    dataset_name, knowledge_level, target_class
)
repalced_model_name = "{}_{}_{:d}_{}_Poisoning_Model.pth".format(
    dataset_name, knowledge_level, target_class, agr_name
)
logger_name = "{}_{}_{:d}_{}_Poisoning".format(
    dataset_name, knowledge_level, target_class, agr_name
)
runtime_logger_name = "{}_{}_{:d}_{}_Poisoning_Running".format(
    dataset_name, knowledge_level, target_class, agr_name
)
poi_index = [0] * (client_num - malicious_num) + [1] * malicious_num
# Please do not change lr
server = Server(Cifar100_model_generator, optim.SGD, {"lr": 0.01})
clients = [
    Client(Cifar100_model_generator, optim.SGD, {"lr": 0.01}) for _ in range(client_num)
]
server.load_model(model_save_path + normal_model_name)
for index, client in enumerate(clients):
    client.load_model(model_save_path + normal_model_name)
general_datasets = sampling_datasets(
    Cifar100_train_dataset, client_num - malicious_num, label_range, each_class_num
)
no_target_datasets = sampling_no_target_class_datasets(
    Cifar100_train_dataset, malicious_num, target_class, dataset_size
)
train_loaders = [
    torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for train_dataset in general_datasets
]
train_loaders += [
    torch.utils.data.DataLoader(no_target_dataset, batch_size=batch_size, shuffle=True)
    for no_target_dataset in no_target_datasets
]
maintask_Cifar100_dataset = remove_target_class_dataset(
    Cifar100_train_dataset, target_class, dcopy=False, transform=None
)
maintask_loader = torch.utils.data.DataLoader(
    maintask_Cifar100_dataset, batch_size, shuffle=False
)
entire_train_loader = Cifar100_train_loader
# Malcicious Model
replacement_state_dict = torch.load(model_save_path + malicious_model_name)
logger = get_logger("./log/S2/" + logger_name + ".log", verbosity=1, name=logger_name)
runtime_logger = get_logger(
    "./log/S2/" + runtime_logger_name + ".log", verbosity=1, name=runtime_logger_name
)
set_agr_for_server(
    agr_name,
    server,
    dataset_name,
    test_dataset_dict,
    runtime_logger,
    malicious_num,
    mal_client_idx=list(range(client_num - malicious_num, client_num)),
    ifprint=False,
)
try:
    for t in range(10):
        input = [next(iter(train_loader)) for train_loader in train_loaders]
        gradient_list, inference_gradient_AGR, previous_global_model_state_dict = (
            fedSGD_batch_gradient_inference_under_AGR(
                server, clients, input, return_candidx=False
            )
        )
        train_acc, train_loss, time_elapsed = (
            fedSGD_epoch_model_replacement_against_defence(
                server,
                clients,
                train_loaders,
                replacement_state_dict,
                inference_gradient_AGR,
                target_class,
                poi_index=poi_index,
                logger=runtime_logger,
                scale=100,
                optimization_round=1000,
                coeff=[1, 0, 0, 0],
                threshold=[8e3, 1e3, 1e4],
                init_value=[10, 1],
                lr=1e-5,
                opt_parameter="both",
                ifprint=False,
            )
        )

        model = server.global_model

        ref_server = Server(Cifar100_model_generator, optim.SGD, {"lr": 0.01})
        ref_server.load_model(model_save_path + malicious_model_name)
        malicious_model = ref_server.global_model

        model_distance = cal_model_distance(model, malicious_model)

        maintask_acc, maintask_loss = epoch_test(maintask_loader, model)
        target_acc, target_loss = epoch_target2(
            entire_train_loader, model, target_class
        )

        print("-------------Epoch: {}--------------".format(t))
        print("Model Distance: {:.6f}".format(model_distance))
        print(
            "Model Train Acc: {:.6f}, Model Train Loss: {:.6f}".format(
                train_acc, train_loss
            )
        )
        print(
            "Model Accuracy on Main Task: {:.6f}, Model Loss on Main Task: {:.6f}".format(
                maintask_acc, maintask_loss
            )
        )
        print(
            "Model Accuracy on Target Class: {:.6f}, Model Loss on Target Class: {:.6f}".format(
                target_acc, target_loss
            )
        )
        print(
            "Epoch complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        runtime_logger.info("-------------Epoch: {}--------------".format(t))
        runtime_logger.info("Model Distance: {:.6f}".format(model_distance))
        runtime_logger.info(
            "Model Train Acc: {:.6f}, Model Train Loss: {:.6f}".format(
                train_acc, train_loss
            )
        )
        runtime_logger.info(
            "Model Accuracy on Main Task: {:.6f}, Model Loss on Main Task: {:.6f}".format(
                maintask_acc, maintask_loss
            )
        )
        runtime_logger.info(
            "Model Accuracy on Target Class: {:.6f}, Model Loss on Target Class: {:.6f}".format(
                target_acc, target_loss
            )
        )
        runtime_logger.info(
            "Epoch complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        logger.info("-------------Epoch: {}--------------".format(t))
        logger.info("Model Distance: {:.6f}".format(model_distance))
        logger.info(
            "Model Train Acc: {:.6f}, Model Train Loss: {:.6f}".format(
                train_acc, train_loss
            )
        )
        logger.info(
            "Model Accuracy on Main Task: {:.6f}, Model Loss on Main Task: {:.6f}".format(
                maintask_acc, maintask_loss
            )
        )
        logger.info(
            "Model Accuracy on Target Class: {:.6f}, Model Loss on Target Class: {:.6f}".format(
                target_acc, target_loss
            )
        )
        logger.info(
            "Epoch complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )


except KeyboardInterrupt:
    print("Stopping")
model = server.global_model

ref_server = Server(Cifar100_model_generator, optim.SGD, {"lr": 0.01})
ref_server.load_model(model_save_path + malicious_model_name)
malicious_model = ref_server.global_model

model_distance = cal_model_distance(model, malicious_model)

maintask_acc, maintask_loss = epoch_test(maintask_loader, model)
target_acc, target_loss = epoch_target2(entire_train_loader, model, target_class)

print("Model Distance: {:.6f}".format(model_distance))
print(
    "Model Accuracy on Main Task: {:.6f}, Model Loss on Main Task: {:.6f}".format(
        maintask_acc, maintask_loss
    )
)
print(
    "Model Accuracy on Target Class: {:.6f}, Model Loss on Target Class: {:.6f}".format(
        target_acc, target_loss
    )
)
server.save_model(model_save_path + repalced_model_name)
