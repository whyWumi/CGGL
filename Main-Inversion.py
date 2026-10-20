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
import inversefed
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.utils import download_and_extract_archive
from lpips import lpips
from skimage.metrics import structural_similarity as ssim
from torch.distributed import ReduceOp
import numpy as np
import sys
import os
from PIL import Image
import global_var
import defense

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(8, sci_mode=True)
args = inversefed.options().parse_args()
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

from Main_framework_lib3 import *

#!pip install lpips
# Dataset
dm = torch.as_tensor(
    getattr(inversefed.consts, f"{args.dataset.lower()}_mean"), **setup
)[:, None, None]
ds = torch.as_tensor(
    getattr(inversefed.consts, f"{args.dataset.lower()}_std"), **setup
)[:, None, None]
# transform1 = transforms.Compose([
#             transforms.Resize(112),
#             transforms.CenterCrop(112),
#             transforms.ToTensor(),
#             transforms.Normalize(dm, ds)])
# MyImageNet_dataset = MyImageNet(data_dir="dataset/myimage/", transform=transform1)
#
#
#     #test_size = len(dataset) - train_size - val_size
# MyImageNet_train_dataset = CustomSubset(MyImageNet_dataset, np.arange(0.8 * len(MyImageNet_dataset)))
# MyImageNet_test_dataset = CustomSubset(MyImageNet_dataset, np.arange(0.2 * len(MyImageNet_dataset), len(MyImageNet_dataset)))
# MyImageNet_train_loader = torch.utils.data.DataLoader(MyImageNet_train_dataset,batch_size=32,shuffle=True)
# MyImageNet_test_loader = torch.utils.data.DataLoader(MyImageNet_test_dataset,batch_size=32,shuffle=False)

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

test_dataset_dict = {
    # "TinyImageNet":TinyImageNet_test_dataset,
    "Cifar100": Cifar100_test_dataset,
    # "CalTech256":CalTech256_test_dataset
    # "MyImageNet":MyImageNet_test_dataset
}

train_dataset_dict = {
    # "TinyImageNet":TinyImageNet_train_dataset,
    "Cifar100": Cifar100_train_dataset,
    # "CalTech256":CalTech256_train_dataset
    # "MyImageNet":MyImageNet_train_dataset
}
test_loader_dict = {
    # "TinyImageNet":TinyImageNet_test_loader,
    "Cifar100": Cifar100_test_loader,
    # "CalTech256":CalTech256_test_loader
    # "MyImageNet":MyImageNet_test_loader
}

train_loader_dict = {
    # "TinyImageNet":TinyImageNet_train_loader,
    "Cifar100": Cifar100_train_loader,
    # "CalTech256":CalTech256_train_loader
    # "MyImageNet":MyImageNet_train_loader
}
data_info = {
    "test_dataset_dict": test_dataset_dict,
    "train_dataset_dict": train_dataset_dict,
    "test_loader_dict": test_loader_dict,
    "train_loader_dict": train_loader_dict,
}
device = torch.device("cuda:0")
# target_class = 1
# output_size = 1


class BNForwardFeatureHook:

    def __init__(self, module, process_group=None):
        self.process_group = process_group
        self.hook = module.register_forwad_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        x = input[0]
        with torch.no_grad():
            channel_first_input = x.transpose(0, 1).contiguous()
            squashed_input_tensor_view = channel_first_input.view(
                channel_first_input.size(0), -1
            )
            local_mean = torch.mean(squashed_input_tensor_view, 1)
            local_sqr_mean = torch.pow(squashed_input_tensor_view, 2).mean(1)
            self.mean = local_mean
            self.var = local_sqr_mean - local_mean.pow(2)

    def close(self):
        self.hook.remove()


class BNForwardLossHook:

    def __init__(self, module, bn_mean, bn_var, process_group=None):

        self.process_group = process_group
        self.bn_mean = bn_mean
        self.bn_var = bn_var
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        mean, var = self.compute_normal_batchnorm_statistics(input[0])

        self.bn_loss = torch.norm(self.bn_mean - mean, 2) + torch.norm(
            self.bn_var - var, 2
        )

        # must have no output

    def compute_normal_batchnorm_statistics(self, x):

        channel_first_input = x.transpose(0, 1).contiguous()
        squashed_input_tensor_view = channel_first_input.view(
            channel_first_input.size(0), -1
        )
        local_mean = torch.mean(squashed_input_tensor_view, 1)
        local_sqr_mean = torch.pow(squashed_input_tensor_view, 2).mean(1)
        mean = local_mean
        var = local_sqr_mean - local_mean.pow(2)

        return mean, var

    def compute_sync_batchnorm_statistics(self, x):
        """Code referenced from https://github.com/NVIDIA/apex/"""

        process_group = self.process_group
        world_size = 1
        if not self.process_group:
            process_group = torch.distributed.group.WORLD

        channel_first_input = x.transpose(0, 1).contiguous()
        squashed_input_tensor_view = channel_first_input.view(
            channel_first_input.size(0), -1
        )

        local_mean = torch.mean(squashed_input_tensor_view, 1)
        local_sqr_mean = torch.pow(squashed_input_tensor_view, 2).mean(1)
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size(process_group)
            torch.distributed.all_reduce(local_mean, ReduceOp.SUM, process_group)
            mean = local_mean / world_size
            torch.distributed.all_reduce(local_sqr_mean, ReduceOp.SUM, process_group)
            sqr_mean = local_sqr_mean / world_size
        else:
            raise ValueError("distributed not initialize")

        var = sqr_mean - mean.pow(2)

        return mean, var

    def close(self):
        self.hook.remove()


def set_BN_regularization(bn_mean_list, bn_var_list, model, args):
    bn_loss_layers = []
    i_bn_layers = 0
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if bn_mean_list is not None and args.exact_bn:
                mean = bn_mean_list[i_bn_layers].detach().clone().cuda(args.gpu)
                var = bn_var_list[i_bn_layers].detach().clone().cuda(args.gpu)
            else:
                mean = module.running_mean.detach().clone()
                var = module.running_var.detach().clone()
            bn_loss_layers.append(BNForwardLossHook(module, mean, var))
            if i_bn_layers == 0:
                print(
                    f"set loss hook for \
                    bn statistics,exact_bn [{args.exact_bn}]"
                )
            i_bn_layers += 1

    return bn_loss_layers


def calc_ssim(target, recon):
    """tructural similarity index"""
    multichannel = True
    tt = transforms.ToPILImage()
    target1 = utils.make_grid(target)
    recon1 = utils.make_grid(recon)
    target_np = np.array(tt(target1))
    recon_np = np.array(tt(recon1))
    # target = target.permute(0, 2, 3, 1).cpu()
    # print(target.shape)c
    # target = target.mul(255).byte()
    # target_np = target.numpy()
    # recon = recon.permute(0, 2, 3, 1).cpu()
    # recon = recon.mul(255).byte()
    # recon_np = recon.numpy()
    ssim_score = ssim(target_np, recon_np, data_range=255, multichannel=multichannel)

    return ssim_score


def calculate_gradient(optimizer, model, dummy_data, dummy_label, gradient_list):
    optimizer.zero_grad()
    model.zero_grad()
    # model.to(device)
    dummy_data = dummy_data  # .to(device)
    dummy_pred = model(dummy_data)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(dummy_pred, dummy_label)  # [0:1].long()
    dummy_dL_dW = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    dL_dW = gradient_list
    loss_grad = 0.0
    pnorm = [0.0, 0.0]
    for gx, gy in zip(dummy_dL_dW, dL_dW):
        loss_grad -= (gx * gy).sum()
        pnorm[0] += gx.pow(2).sum()
        pnorm[1] += gy.pow(2).sum()

    loss_grad = 1 + loss_grad / pnorm[0].sqrt() / pnorm[1].sqrt()
    return loss_grad


def get_best_pseudo(loss_list, x_pseudo_list, y_pseudo, gradient_list):  # y_pseudo
    # get better x_pseudo
    loss_list = torch.as_tensor(loss_list)
    min_loss_idx = loss_list.argmin()
    max_loss_idx = loss_list.argmax()
    x_pseudo = x_pseudo_list[min_loss_idx].detach().clone().cuda()
    max_pseudo = x_pseudo_list[max_loss_idx].detach().clone().cuda()
    optimizer = torch.optim.Adam([x_pseudo, y_pseudo], lr=1e-1)
    maxoptimizer = torch.optim.Adam([max_pseudo, y_pseudo], lr=1e-1)
    # dummy_pred = model(dummy_data)
    max_loss = calculate_gradient(optimizer, model, max_pseudo, y_pseudo, gradient_list)
    min_loss = calculate_gradient(optimizer, model, x_pseudo, y_pseudo, gradient_list)
    # max_loss = rec_machine._gradient_closure(maxoptimizer, max_pseudo, input_gradient, y_pseudo)().item()
    # min_loss = rec_machine._gradient_closure(optimizer, x_pseudo, input_gradient, y_pseudo)().item()
    print(f"orig loss {min_loss}")
    print(f"max loss {max_loss}")
    n_seed = args.n_seed
    batch_size = 1  ###args.num_images

    rank = 0
    max_loss = calculate_gradient(optimizer, model, max_pseudo, y_pseudo, gradient_list)
    min_loss = calculate_gradient(optimizer, model, x_pseudo, y_pseudo, gradient_list)
    for i_samples in range(batch_size):
        x_test = x_pseudo.detach().clone()
        best_seed = min_loss_idx

        for i_seed in range(n_seed):
            x_test[i_samples] = x_pseudo_list[i_seed][i_samples]
            loss = calculate_gradient(
                optimizer, model, x_test.cuda(), y_pseudo, gradient_list
            )
            if loss < min_loss:
                min_loss = loss
                best_seed = i_seed

        x_pseudo[i_samples] = x_pseudo_list[best_seed][i_samples].detach().clone()
        k = i_samples + rank * batch_size
        print(f"i_imgs: {k} best_seed:{best_seed} loss: {min_loss}")

    for i_samples in range(batch_size):
        x_test = max_pseudo.detach().clone()
        best_seed = max_loss_idx

        for i_seed in range(n_seed):
            x_test[i_samples] = x_pseudo_list[i_seed][i_samples]
            loss = calculate_gradient(
                maxoptimizer, model, x_test.cuda(), y_pseudo, gradient_list
            )
            if loss > max_loss:
                max_loss = loss
                best_seed = i_seed

        max_pseudo[i_samples] = x_pseudo_list[best_seed][i_samples].detach().clone()
        k = i_samples + rank * batch_size
        print(f"i_imgs: {k} best_seed:{best_seed} loss: {min_loss}")
    max_loss = calculate_gradient(
        maxoptimizer, model, x_pseudo.cuda(), y_pseudo, gradient_list
    )
    print(f"max loss: {max_loss}")
    return x_pseudo.detach().clone(), max_pseudo.detach().clone()


def get_y_pseudo(args, target_gradient, labels):

    label_pred = (
        dgi_label_recon(target_gradient, args.num_images, args.n_classes)
        .detach()
        .view(-1, 1)
    )
    print(f"[dgi info] y pred: {label_pred.view(-1,).cpu().numpy()}")

    n_correct, acc = compute_label_acc(labels, label_pred)
    print(f" > label acc: {acc * 100:.2f}%")

    return label_pred.view(
        -1,
    )


def cal_gradient(model, X, y):
    model.eval()

    X, y = X.to(device), y.to(device)
    yp = model(X)
    y = y.to(torch.int64)
    loss = nn.CrossEntropyLoss()(yp, y)

    gradient = torch.autograd.grad(loss, model.parameters())
    gradient = [grad for grad in gradient]

    return gradient


##Exp
settings = [
    (
        {
            "batch_size": 8,  ##16 8-64 4
            "client_num": 40,  ##40
            "malicious_num": 10,  # 10
        },
        0,
        "Cifar100",
        "FK",
        "MKrum",  ##SK  MKrum Standard
        {
            "optimization_round": 1000,
            "threshold": [8e3, 1e2, 3e4],
            "init_value": [1, 0],
            "lr": 1e-7,
            "opt_parameter": "gamma",
        },
        False,
    )
]
(
    fl_setting,
    target_class,
    dataset_name,
    knowledge_level,
    agr_name,
    agr_attack_setting_dict,
    sp,
) = settings[0]
gradient_tuple = gradient_simulate(
    dataset_name,
    data_info,
    target_class,
    knowledge_level,
    agr_name,
    fl_setting,
    agr_attack_setting_dict,
    sp,
)

#
# loss_list = []
# x_pseudo_list = []

(
    target_img_batch,
    gradient_list,
    inference_gradient,
    previous_global_model_state_dict,
) = gradient_tuple

X, y = target_img_batch
ground_truth = X.to("cuda")
# ground_truth = X[0:1].to('cuda')
y = y.to(torch.long)
# y = y[0:1].to(torch.long)
labels = y.to("cuda")
torchvision.utils.save_image(X, os.path.join(args.image_path, "x.png"))


original_image = X[0:1]
vutils.save_image(
    original_image.reshape([1, 3, 32, 32]),
    "./result/"
    + dataset_name
    + "_"
    + knowledge_level
    + "_"
    + str(target_class)
    + "_"
    + agr_name
    + "_Original.png",
    normalize=False,
    scale_each=True,
    nrow=int(4),
)


model = (
    Cifar100_model_generator()
)  # Cifar100_model_generator MyImageNet_model_generator
model.load_state_dict(previous_global_model_state_dict)

inversion_logger = get_logger(
    "./log/S3/"
    + dataset_name
    + "_"
    + knowledge_level
    + "_"
    + str(target_class)
    + "_"
    + agr_name
    + "_Inversion.log",
    verbosity=1,
    name=dataset_name
    + "_"
    + knowledge_level
    + "_"
    + str(target_class)
    + "_"
    + agr_name
    + "_Inversion",
)

save_name = (
    dataset_name + "_" + knowledge_level + "_" + str(target_class) + "_" + agr_name
)
inversion_args = [1e-1, 1e-2, 10, 0.001]

x_group = None
# inference_gradient = cal_gradient(model,X,y)
input_gradient = inference_gradient


# BN
if args.upload_bn:
    bn_mean_var_layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_mean_var_layers.append(module)
            # bn_mean_var_layers.append(BNForwardFeatureHook(
            #     module))

bn_mean_list = []
bn_var_list = []
if args.upload_bn:
    for mod in bn_mean_var_layers:
        bn_mean_list.append(mod.running_mean)
        bn_var_list.append(mod.running_var)
    # bn_mean_list = [mod.running_mean.clone() for mod in bn_mean_var_layers]
    # bn_var_list = [mod.running_var.clone() for mod in bn_mean_var_layers]

    # bn_mean_list = [mod.mean.detach().clone() for mod in bn_mean_var_layers]
    # bn_var_list = [mod.var.detach().clone() for mod in bn_mean_var_layers]

bn_loss_layers = None
if args.BN > 0:
    bn_loss_layers = set_BN_regularization(bn_mean_list, bn_var_list, model, args)


img_shape = [3, 32, 32]

if args.optim == "ours":
    config = dict(
        signed=args.signed,
        boxed=args.boxed,
        cost_fn=args.cost_fn,
        indices="def",
        weights="equal",
        lr=0.1,  ##0,1 0.1046 1e-1
        optim=args.optimizer,
        restarts=args.restarts,
        max_iterations=12000,
        total_variation=args.tv,
        init="randn",
        filter="none",
        lr_decay=True,
        scoring_choice="loss",
        BN=args.BN,
        Group=args.Group,
        Clip=args.clip,
        Scale=args.scale,
    )
##标签推断
if args.inferlabel:
    pseudo_label = get_y_pseudo(args, input_gradient, labels).to(device)
else:
    pseudo_label = labels
rec_machine = inversefed.GradientReconstructor(
    model, (dm, ds), config, num_images=args.num_images, bn_loss_layers=bn_loss_layers
)
loss_list = []
x_pseudo_list = []

for i in range(args.n_seed):
    output, stats = rec_machine.reconstruct(
        ground_truth,
        input_gradient,
        pseudo_label,
        img_shape=img_shape,
        dryrun=args.dryrun,
    )
    loss_list.append(stats["opt"])
    x_pseudo_list.append(output)

if args.n_seed != 0:
    x_group, x_max = get_best_pseudo(
        loss_list, x_pseudo_list, pseudo_label, input_gradient
    )
    print(x_group)
else:
    x_group, x_max = None, None

##防御
if args.defense_method == "None":
    print("No defense applied.")
    # d_param = config['defense_setting']
else:
    if args.defense_method == "noise":
        d_param = 0.01  # if config['defense_setting']['noise'] is None else config['defense_setting']['noise']
        input_gradient = defense.additive_noise(input_gradient, std=d_param)
    if args.defense_method == "clipping":
        d_param = 4  # 4if config['defense_setting']['clipping'] is None else config['defense_setting']['clipping']
        input_gradient = defense.gradient_clipping(input_gradient, bound=d_param)
    if args.defense_method == "compression":
        d_param = 20  # 20if config['defense_setting']['compression'] is None else config['defense_setting'][
        # 'compression']
        input_gradient = defense.gradient_compression(
            input_gradient, percentage=d_param
        )
    if args.defense_method == "representation":
        d_param = 10  # 10if config['defense_setting']['representation'] is None else config['defense_setting'][
        # 'representation']
        input_gradient = defense.perturb_representation(
            input_gradient, model, ground_truth, pruning_rate=d_param
        )
    # else:
    #     raise NotImplementedError("Invalid defense method!")
    print("Defense applied: {} w/ {}.".format(args.defense_method, d_param))


finalrec_machine = inversefed.GradientReconstructor(
    model,
    (dm, ds),
    config,
    num_images=args.num_images,
    bn_loss_layers=bn_loss_layers,
    x_group=x_group,
    x_max=x_max,
)
output, stats = finalrec_machine.reconstruct(
    ground_truth, input_gradient, pseudo_label, img_shape=img_shape, dryrun=args.dryrun
)

# Compute stats

lpips_loss = lpips.LPIPS(net="vgg", spatial=False).to(device)
lpips_score = 99.0
# score = lpips_loss(output, ground_truth).squeeze().item()
# if score < lpips_score:
#         lpips_score = score
for i in range(args.num_images):
    score = lpips_loss(output[i], ground_truth[i]).squeeze().item()
    if score < lpips_score:
        lpips_score = score


output_denormalized1 = torch.clamp(output * ds + dm, 0, 1)
ground_truth_denormalized1 = torch.clamp(ground_truth * ds + dm, 0, 1)
# ground_truth1 = (ground_truth-0.5).true_divide(0.5)
# ssim_score = calc_ssim(output, ground_truth)
ssim_score = calc_ssim(output_denormalized1, ground_truth_denormalized1)

test_mse = (output - ground_truth).pow(2).mean().item()
feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1 / ds)

# Save the resulting image
if args.save_image and not args.dryrun:
    os.makedirs(args.image_path, exist_ok=True)
    output_denormalized = torch.clamp(output * ds + dm, 0, 1)

    if args.num_images == 1:
        if args.dataset == "myimagenet":
            rec_filename = f"{args.model}trained_imagenet_newb1_df_{args.cost_fn}-{args.target_id}.png"
        elif args.dataset == "cifar100":
            rec_filename = f"{args.model}trained_cifar100_b1_df_{args.cost_fn}-{args.target_id}.png"
        # else:
        #     rec_filename = (
        #         f'{validloader.dataset.classes[labels][0]}_{"trained" if args.trained_model else ""}'
        #         f"{args.model}_{args.cost_fn}-{args.target_id}.png"
        #     )
    else:
        rec_filename = (
            f"{args.model}_normalized_{args.defense_method}-{args.cost_fn}.png"
        )
        torchvision.utils.save_image(
            output, os.path.join(args.image_path, rec_filename)
        )  ##保存整组图像的 Res18_none_sim.png_
    index = ["1", "2", "3", "4", "5", "6", "7", "8"]
    # ##保存单张图片
    for i in range(8):
        torchvision.utils.save_image(
            output[i],
            os.path.join(args.image_path, index[i] + f"_normalized_myrecon.png"),
        )  # !!!

    gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
    if args.num_images == 1:
        if args.dataset == "myimagenet":
            gt_filename = f"1_ground_truth-{args.target_id}.png"
        elif args.dataset == "cifar100":
            gt_filename = (
                f"{args.model}ground_truth_cifar100_{args.cost_fn}-{args.target_id}.png"
            )
        # else:
        #     gt_filename = f"{validloader.dataset.classes[labels][0]}_ground_truth-{args.target_id}.png"
    else:
        gt_filename = f"32cifar_ground_truth.png"

    # gt_filename = f"{validloader.dataset.classes[labels][0]}_ground_truth-{args.target_id}.png"
    torchvision.utils.save_image(
        ground_truth, os.path.join(args.image_path, gt_filename)
    )
    for i in range(8):
        torchvision.utils.save_image(
            ground_truth[i], os.path.join(args.image_path, index[i] + f"groundtrue.png")
        )  # !!!

else:
    rec_filename = None
    gt_filename = None

# Save to a table:
print(
    f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} | lpips_score: {lpips_score:2.4e} | ssim_score: {ssim_score:2.4e} "
)
