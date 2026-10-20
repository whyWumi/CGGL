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

from dataset import *
from common_DL import *
from gradient_lib import *
from federated_learning import *
from model_structure import *
from utils import *
from inversion_attacks import *
from model_structure import *

from Main_framework_lib1 import *

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
#
# CalTech256_train_loader = torch.utils.data.DataLoader(CalTech256_train_dataset,batch_size=16,shuffle=True)
# CalTech256_test_loader = torch.utils.data.DataLoader(CalTech256_test_dataset,batch_size=16,shuffle=False)

##Cifar100
##Cifar100 Federated Learning

lr = 1e-2
client_num = 50
epoch_num = 30
batch_size = 128
each_dataset_size = 30000
server = Server(Cifar100_model_generator, optim.SGD, {"lr": lr})
clients = [
    Client(Cifar100_model_generator, optim.SGD, {"lr": lr}) for _ in range(client_num)
]
Cifar100_train_datasets = sampling_datasets_iid(
    Cifar100_train_dataset, client_num, each_dataset_size
)
train_loaders = [
    torch.utils.data.DataLoader(
        Cifar100_train_dataset,
        batch_size,
        shuffle=True,
    )
    for train_dataset in Cifar100_train_datasets
]
# try:
#     for t in range(epoch_num):
#         train_acc,train_loss,test_acc,test_loss,time_elapsed = fedSGD_epoch(server,clients,train_loaders,Cifar100_test_loader)
#
#         print("-------------Epoch: %d--------------" % t)
#         print("Train_Acc: {:.6f} ,Test_Acc: {:.6f}".format(train_acc, test_acc))
#         print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#
# except KeyboardInterrupt:
#     print("Stopping")
# server.save_model("./model/Cifar100_Normal_Model.pth")
server.load_model("./model/Cifar100_Normal_Model.pth")

# Cifar100 Full Knowledge
target_class = 0
dataset_size = 100000
batch_size = 128
dataset_name = "Cifar100"
client = Client(Cifar100_model_generator, optim.SGD, {"lr": 0.01})
no_target_Cifar100_dataset = sampling_no_target_class_dataset(
    Cifar100_train_dataset, target_class, dataset_size, dcopy=False
)
no_target_Cifar100_train_loader = torch.utils.data.DataLoader(
    no_target_Cifar100_dataset, batch_size, shuffle=True
)
maintask_Cifar100_dataset = remove_target_class_dataset(
    Cifar100_train_dataset, target_class, dcopy=False, transform=None
)
maintask_Cifar100_loader = torch.utils.data.DataLoader(
    maintask_Cifar100_dataset, batch_size, shuffle=False
)
hyperparameter_dict = {
    "main_task_lr": 1e-2,
    "main_task_lr_decay": True,
    "main_task_lr_decay_gamma": 0.1,
    "main_task_lr_decay_milestone": [15],
    "malicious_task_lr": 2e-5,
    "save_interal": None,
    "epoch_num": 30,
}
logger = get_logger(
    "./log/S1/Cifar100_FK_Malicious_Model.log", name="Cifar100_FK_Malicious_Model"
)
training_malicious_model_with_full_knowledge(
    dataset_name,
    client,
    target_class,
    no_target_Cifar100_train_loader,
    maintask_Cifar100_loader,
    Cifar100_train_loader,
    hyperparameter_dict,
    logger,
    save_path="./model/",
    save_name=None,
    ifprint=True,
)

# #Cifar100 Semi Knowledge
# target_class = 0
# target_class_num = 30
# label_range = 100
# each_class_num = 400
# batch_size = 128
# dataset_name = "Cifar100"
# client = Client(Cifar100_model_generator,optim.SGD,{'lr':0.01})
# client.load_model("./model/Cifar100_Normal_Model.pth")
# only_target_Cifar100_dataset = sampling_only_target_class_dataset(Cifar100_test_dataset, target_class, target_class_num, label_range, dcopy=False, transform=None)
# only_target_Cifar100_train_loader = torch.utils.data.DataLoader(only_target_Cifar100_dataset,batch_size,shuffle=True)
# label_list = random.sample(list(range(0,target_class))+list(range(target_class+1,label_range)),k=30)
# print(label_list)
# local_Cifar100_dataset = sampling_no_target_class_dataset_niid(Cifar100_train_dataset, target_class, label_list, label_range, each_class_num, dcopy=False, transform=None)
# local_Cifar100_train_loader = torch.utils.data.DataLoader(no_target_Cifar100_dataset,batch_size,shuffle=True)
# maintask_Cifar100_dataset = remove_target_class_dataset(Cifar100_train_dataset, target_class, dcopy=False, transform=None)
# maintask_Cifar100_loader = torch.utils.data.DataLoader(maintask_Cifar100_dataset,batch_size,shuffle=False)
# hyperparameter_dict = {
#                 "main_task_lr" : 1e-3,
#                 "main_task_lr_decay" : False,
#                 "malicious_task_lr" : 2e-2,
#                 "save_interal" : None,
#                 "epoch_num" : 3
#             }
# logger = get_logger("./log/S1/Cifar100_SK_Malicious_Model.log",name="Cifar100_SK_Malicious_Model")
# training_malicious_model_with_semi_knowledge(dataset_name, client, target_class, only_target_Cifar100_train_loader, local_Cifar100_train_loader, maintask_Cifar100_loader, Cifar100_train_loader, hyperparameter_dict, logger,
#                                                  save_path="./model/", save_name=None, ifprint=True)
#
# #Cifar100 No Knowledge
# target_class = 0
# label_range = 100
# inversion_data_size = 256
# each_class_num = 400
# batch_size = 128
# loader_size = 50
# dataset_name = "Cifar100"
# client = Client(Cifar100_model_generator,optim.SGD,{'lr':0.01})
# client.load_model("./model/Cifar100_Normal_Model.pth")
# logger = get_logger("./log/S1/Cifar100_NK_MI.log",name="Cifar100_NK_MI")
# model = client.client_model
#
# history,inversion_data = deep_inversion(model,inversion_data_size,target_class,logger,save_name="Cifar100_MI_data_"+str(target_class),
#                    epoch_num=5000,
#                    main_coeff=1e-1,
#                    l2_coeff=1e-5,
#                    tv_coeff=1e-3,
#                    bn_coeff=1,
#                    first_bn_weight=10,
#                    lr=0.25,
#                    image_size=32,
#                    save_path="./temp_data/",
#                    ifprint=True,
#                    ifhistory=False)
# tensor_data = torch.load("./temp_data/Cifar100_MI_data_"+str(target_class)+".tensor")
# only_inversion_target_Cifar100_train_loader =  tensor2loader(tensor_data,target_class,batch_size,loader_size)
# label_list = random.sample(list(range(0,target_class))+list(range(target_class+1,label_range)),k=30)
# print(label_list)
# local_Cifar100_dataset = sampling_no_target_class_dataset_niid(Cifar100_train_dataset, target_class, label_list, label_range, each_class_num, dcopy=False, transform=None)
# local_Cifar100_train_loader = torch.utils.data.DataLoader(no_target_Cifar100_dataset,batch_size,shuffle=True)
# maintask_Cifar100_dataset = remove_target_class_dataset(Cifar100_train_dataset, target_class, dcopy=False, transform=None)
# maintask_Cifar100_loader = torch.utils.data.DataLoader(maintask_Cifar100_dataset,batch_size,shuffle=False)
# hyperparameter_dict = {
#                 "main_task_lr" : 1e-3,
#                 "main_task_lr_decay" : False,
#                 "malicious_task_lr" : 1e-3,
#                 "save_interal" : None,
#                 "epoch_num" : 2
#             }
# logger = get_logger("./log/S1/Cifar100_NK_Malicious_Model.log",name="Cifar100_NK_Malicious_Model")
# training_malicious_model_with_no_knowledge(dataset_name, client, target_class, only_inversion_target_Cifar100_train_loader, local_Cifar100_train_loader, maintask_Cifar100_loader, Cifar100_train_loader,
#                                            hyperparameter_dict, logger, save_path="./model/", save_name=None, ifprint=True)
#
# #TinyImageNet
# #TinyImageNet Federated Learning
# lr = 1e-2
# client_num = 40
# epoch_num = 30
# batch_size = 128
# each_dataset_size = 10000
# server = Server(TinyImageNet_model_generator,optim.SGD,{'lr':lr})
# clients = [Client(TinyImageNet_model_generator,optim.SGD,{'lr':lr}) for _ in range(client_num)]
# TinyImageNet_train_datasets = sampling_datasets_iid(TinyImageNet_train_dataset,client_num,each_dataset_size)
# train_loaders = [torch.utils.data.DataLoader(TinyImageNet_train_dataset,batch_size,shuffle=True) for train_dataset in TinyImageNet_train_datasets]
# try:
#     for t in range(epoch_num):
#         train_acc,train_loss,test_acc,test_loss,time_elapsed = fedSGD_epoch(server,clients,train_loaders,TinyImageNet_test_loader)
#
#         print("-------------Epoch: %d--------------" % t)
#         print("Train_Acc: {:.6f} ,Test_Acc: {:.6f}".format(train_acc, test_acc))
#         print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#
# except KeyboardInterrupt:
#     print("Stopping")
# server.save_model("./model/TinyImageNet_Normal_Model.pth")
# server.load_model("./model/TinyImageNet_Normal_Model.pth")
# #TinyImageNet Full Knowledge
# target_class = 0
# dataset_size = 300000
# batch_size = 128
# dataset_name = "TinyImageNet"
# client = Client(TinyImageNet_model_generator,optim.SGD,{'lr':0.01})
# no_target_TinyImageNet_dataset = sampling_no_target_class_dataset(TinyImageNet_train_dataset,target_class,dataset_size,dcopy=False)
# no_target_TinyImageNet_train_loader = torch.utils.data.DataLoader(no_target_TinyImageNet_dataset,batch_size,shuffle=True)
# maintask_TinyImageNet_dataset = remove_target_class_dataset(TinyImageNet_train_dataset, target_class, dcopy=False, transform=None)
# maintask_TinyImageNet_loader = torch.utils.data.DataLoader(maintask_TinyImageNet_dataset,batch_size,shuffle=False)
# hyperparameter_dict = {
#                 "main_task_lr" : 1e-2,
#                 "main_task_lr_decay" : True,
#                 "main_task_lr_decay_gamma" : 0.1,
#                 "main_task_lr_decay_milestone" : [15],
#                 "malicious_task_lr" : 4e-5,
#                 "save_interal" : None,
#                 "epoch_num" : 30
#             }
# logger = get_logger("./log/S1/TinyImageNet_FK_Malicious_Model.log",name="TinyImageNet_FK_Malicious_Model")
# training_malicious_model_with_full_knowledge(dataset_name, client, target_class, no_target_TinyImageNet_train_loader, maintask_TinyImageNet_loader, TinyImageNet_train_loader, hyperparameter_dict, logger,
#                                                  save_path="./model/", save_name=None, ifprint=True)
#
# #TinyImageNet Semi Knowledge
# target_class = 0
# target_class_num = 30
# label_range = 200
# each_class_num = 400
# batch_size = 128
# dataset_name = "TinyImageNet"
# client = Client(TinyImageNet_model_generator,optim.SGD,{'lr':0.01})
# client.load_model("./model/TinyImageNet_Normal_Model.pth")
# only_target_TinyImageNet_dataset = sampling_only_target_class_dataset(TinyImageNet_test_dataset, target_class, target_class_num, label_range, dcopy=False, transform=None)
# only_target_TinyImageNet_train_loader = torch.utils.data.DataLoader(only_target_TinyImageNet_dataset,batch_size,shuffle=True)
# label_list = random.sample(list(range(0,target_class))+list(range(target_class+1,label_range)),k=30)
# print(label_list)
# local_TinyImageNet_dataset = sampling_no_target_class_dataset_niid(TinyImageNet_train_dataset, target_class, label_list, label_range, each_class_num, dcopy=False, transform=None)
# local_TinyImageNet_train_loader = torch.utils.data.DataLoader(no_target_TinyImageNet_dataset,batch_size,shuffle=True)
# maintask_TinyImageNet_dataset = remove_target_class_dataset(TinyImageNet_train_dataset, target_class, dcopy=False, transform=None)
# maintask_TinyImageNet_loader = torch.utils.data.DataLoader(maintask_TinyImageNet_dataset,batch_size,shuffle=False)
# hyperparameter_dict = {
#                 "main_task_lr" : 1e-3,
#                 "main_task_lr_decay" : False,
#                 "malicious_task_lr" : 3e-2,
#                 "save_interal" : None,
#                 "epoch_num" : 3
#             }
# logger = get_logger("./log/S1/TinyImageNet_SK_Malicious_Model.log",name="TinyImageNet_SK_Malicious_Model")
# training_malicious_model_with_semi_knowledge(dataset_name, client, target_class, only_target_TinyImageNet_train_loader, local_TinyImageNet_train_loader, maintask_TinyImageNet_loader, TinyImageNet_train_loader, hyperparameter_dict, logger,
#                                                  save_path="./model/", save_name=None, ifprint=True)
# #TinyImageNet No Knowledge
# target_class = 0
# label_range = 200
# inversion_data_size = 256
# each_class_num = 400
# batch_size = 128
# loader_size = 50
# dataset_name = "TinyImageNet"
# client = Client(TinyImageNet_model_generator,optim.SGD,{'lr':0.01})
# client.load_model("./model/TinyImageNet_Normal_Model.pth")
# logger = get_logger("./log/S1/TinyImageNet_NK_MI.log",name="TinyImageNet_NK_MI")
# model = client.client_model
#
# history,inversion_data = deep_inversion(model,inversion_data_size,target_class,logger,save_name="TinyImageNet_MI_data_"+str(target_class),
#                    epoch_num=5000,
#                    main_coeff=1e-1,
#                    l2_coeff=1e-5,
#                    tv_coeff=1e-3,
#                    bn_coeff=1,
#                    first_bn_weight=10,
#                    lr=0.25,
#                    image_size=64,
#                    save_path="./temp_data/",
#                    ifprint=True,
#                    ifhistory=False)
# tensor_data = torch.load("./temp_data/TinyImageNet_MI_data_"+str(target_class)+".tensor")
# only_inversion_target_TinyImageNet_train_loader =  tensor2loader(tensor_data,target_class,batch_size,loader_size)
# label_list = random.sample(list(range(0,target_class))+list(range(target_class+1,label_range)),k=60)
# print(label_list)
# local_TinyImageNet_dataset = sampling_no_target_class_dataset_niid(TinyImageNet_train_dataset, target_class, label_list, label_range, each_class_num, dcopy=False, transform=None)
# local_TinyImageNet_train_loader = torch.utils.data.DataLoader(no_target_TinyImageNet_dataset,batch_size,shuffle=True)
# maintask_TinyImageNet_dataset = remove_target_class_dataset(TinyImageNet_train_dataset, target_class, dcopy=False, transform=None)
# maintask_TinyImageNet_loader = torch.utils.data.DataLoader(maintask_TinyImageNet_dataset,batch_size,shuffle=False)
# hyperparameter_dict = {
#                 "main_task_lr" : 1e-5,
#                 "main_task_lr_decay" : False,
#                 "malicious_task_lr" : 1e-3,
#                 "save_interal" : None,
#                 "epoch_num" : 2
#             }
# logger = get_logger("./log/S1/TinyImageNet_NK_Malicious_Model.log",name="TinyImageNet_NK_Malicious_Model")
# training_malicious_model_with_no_knowledge(dataset_name, client, target_class, only_inversion_target_TinyImageNet_train_loader, local_TinyImageNet_train_loader, maintask_TinyImageNet_loader, TinyImageNet_train_loader,
#                                            hyperparameter_dict, logger, save_path="./model/", save_name=None, ifprint=True)
#
# #CalTech256
# #CalTech256 Federated Learning
# lr = 1e-2
# client_num = 10
# epoch_num = 30
# batch_size = 16
# each_dataset_size = 30000
# server = Server(CalTech256_model_generator,optim.SGD,{'lr':lr})
# clients = [Client(CalTech256_model_generator,optim.SGD,{'lr':lr}) for _ in range(client_num)]
# CalTech256_train_datasets = sampling_datasets_iid(CalTech256_train_dataset,client_num,each_dataset_size)
# train_loaders = [torch.utils.data.DataLoader(CalTech256_train_dataset,batch_size,shuffle=True) for train_dataset in CalTech256_train_datasets]
# try:
#     for t in range(epoch_num):
#         train_acc,train_loss,test_acc,test_loss,time_elapsed = fedSGD_epoch(server,clients,train_loaders,CalTech256_test_loader)
#
#         print("-------------Epoch: %d--------------" % t)
#         print("Train_Acc: {:.6f} ,Test_Acc: {:.6f}".format(train_acc, test_acc))
#         print("Epoch complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
#
# except KeyboardInterrupt:
#     print("Stopping")
# server.save_model("./model/CalTech256_Normal_Model.pth")
# server.load_model("./model/CalTech256_Normal_Model.pth")
# #CalTech256 Full Knowledge
# target_class = 48
# dataset_size = 50000
# batch_size = 16
# dataset_name = "CalTech256"
# client = Client(CalTech256_model_generator,optim.SGD,{'lr':0.01})
# no_target_CalTech256_dataset = sampling_no_target_class_dataset(CalTech256_train_dataset,target_class,dataset_size,dcopy=False)
# no_target_CalTech256_train_loader = torch.utils.data.DataLoader(no_target_CalTech256_dataset,batch_size,shuffle=True)
# maintask_CalTech256_dataset = remove_target_class_dataset(CalTech256_train_dataset, target_class, dcopy=False, transform=None)
# maintask_CalTech256_loader = torch.utils.data.DataLoader(maintask_CalTech256_dataset,batch_size,shuffle=False)
# hyperparameter_dict = {
#                 "main_task_lr" : 1e-2,
#                 "main_task_lr_decay" : True,
#                 "main_task_lr_decay_gamma" : 0.1,
#                 "main_task_lr_decay_milestone" : [15],
#                 "malicious_task_lr" : 1e-5,
#                 "save_interal" : None,
#                 "epoch_num" : 30
#             }
# logger = get_logger("./log/S1/CalTech256_FK_Malicious_Model.log",name="CalTech256_FK_Malicious_Model")
# training_malicious_model_with_full_knowledge(dataset_name, client, target_class, no_target_CalTech256_train_loader, maintask_CalTech256_loader, CalTech256_train_loader, hyperparameter_dict, logger,
#                                                  save_path="./model/", save_name=None, ifprint=True)
# #CalTech256 Semi Knowledge
# target_class = 48
# target_class_num = 30
# label_range = 256
# each_class_num = 400
# batch_size = 16
# dataset_name = "CalTech256"
# client = Client(CalTech256_model_generator,optim.SGD,{'lr':0.01})
# client.load_model("./model/CalTech256_Normal_Model.pth")
# only_target_CalTech256_dataset = sampling_only_target_class_dataset(CalTech256_test_dataset, target_class, target_class_num, label_range, dcopy=False, transform=None)
# only_target_CalTech256_train_loader = torch.utils.data.DataLoader(only_target_CalTech256_dataset,batch_size,shuffle=True)
# label_list = random.sample(list(range(0,target_class))+list(range(target_class+1,label_range)),k=30)
# print(label_list)
# local_CalTech256_dataset = sampling_no_target_class_dataset_niid(CalTech256_train_dataset, target_class, label_list, label_range, each_class_num, dcopy=False, transform=None)
# local_CalTech256_train_loader = torch.utils.data.DataLoader(no_target_CalTech256_dataset,batch_size,shuffle=True)
# maintask_CalTech256_dataset = remove_target_class_dataset(CalTech256_train_dataset, target_class, dcopy=False, transform=None)
# maintask_CalTech256_loader = torch.utils.data.DataLoader(maintask_CalTech256_dataset,batch_size,shuffle=False)
# hyperparameter_dict = {
#                 "main_task_lr" : 1e-4,
#                 "main_task_lr_decay" : False,
#                 "malicious_task_lr" : 5e-3,
#                 "save_interal" : None,
#                 "epoch_num" : 3
#             }
# logger = get_logger("./log/S1/CalTech256_SK_Malicious_Model.log",name="CalTech256_SK_Malicious_Model")
# training_malicious_model_with_semi_knowledge(dataset_name, client, target_class, only_target_CalTech256_train_loader, local_CalTech256_train_loader, maintask_CalTech256_loader, CalTech256_train_loader, hyperparameter_dict, logger,
#                                                  save_path="./model/", save_name=None, ifprint=True)
# #CalTech256 No Knowledge
# target_class = 48
# label_range = 256
# inversion_data_size = 256
# each_class_num = 400
# batch_size = 16
# loader_size = 50
# dataset_name = "CalTech256"
# client = Client(CalTech256_model_generator,optim.SGD,{'lr':0.01})
# client.load_model("./model/CalTech256_Normal_Model.pth")
# logger = get_logger("./log/S1/CalTech256_NK_MI.log",name="CalTech256_NK_MI")
# model = client.client_model
#
# history,inversion_data = deep_inversion(model,inversion_data_size,target_class,logger,save_name="CalTech256_MI_data_"+str(target_class),
#                    epoch_num=5000,
#                    main_coeff=1e-1,
#                    l2_coeff=1e-5,
#                    tv_coeff=1e-3,
#                    bn_coeff=1,
#                    first_bn_weight=10,
#                    lr=0.25,
#                    image_size=112,
#                    save_path="./temp_data/",
#                    ifprint=True,
#                    ifhistory=False)
# tensor_data = torch.load("./temp_data/CalTech256_MI_data_"+str(target_class)+".tensor")
# only_inversion_target_CalTech256_train_loader =  tensor2loader(tensor_data,target_class,batch_size,loader_size)
# label_list = random.sample(list(range(0,target_class))+list(range(target_class+1,label_range)),k=30)
# print(label_list)
# local_CalTech256_dataset = sampling_no_target_class_dataset_niid(CalTech256_train_dataset, target_class, label_list, label_range, each_class_num, dcopy=False, transform=None)
# local_CalTech256_train_loader = torch.utils.data.DataLoader(no_target_CalTech256_dataset,batch_size,shuffle=True)
# maintask_CalTech256_dataset = remove_target_class_dataset(CalTech256_train_dataset, target_class, dcopy=False, transform=None)
# maintask_CalTech256_loader = torch.utils.data.DataLoader(maintask_CalTech256_dataset,batch_size,shuffle=False)
# hyperparameter_dict = {
#                 "main_task_lr" : 1e-3,
#                 "main_task_lr_decay" : False,
#                 "malicious_task_lr" : 1e-3,
#                 "save_interal" : None,
#                 "epoch_num" : 2
#             }
# logger = get_logger("./log/S1/CalTech256_NK_Malicious_Model.log",name="CalTech256_NK_Malicious_Model")
# training_malicious_model_with_no_knowledge(dataset_name, client, target_class, only_inversion_target_CalTech256_train_loader, local_CalTech256_train_loader, maintask_CalTech256_loader, CalTech256_train_loader,
#                                            hyperparameter_dict, logger, save_path="./model/", save_name=None, ifprint=True)
