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
import argparse
from cggl_paths import (
    CIFAR_DIFFUSION_CHECKPOINT,
    IMAGENET_DIFFUSION_CHECKPOINT,
    IMAGENET_CLASSIFIER_CHECKPOINT,
    IMAGES_DIR,
)
import torch.distributed as dist
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.utils import download_and_extract_archive
import numpy as np
import sys
import os
from PIL import Image

from dataset import *
from common_DL import *
from gradient_lib import *
from federated_learning import *
from model_structure import *
from utils import *
from inversion_attacks import *
from Main_framework_utils import *
from Main_framework_lib2 import *
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


import global_var

device = global_var.get_device()


DEFAULT_CONFIG = dict(
    signed=False,
    boxed=True,
    cost_fn="sim",
    indices="def",
    weights="equal",
    lr=0.1,
    optim="adam",
    restarts=1,
    max_iterations=4800,
    total_variation=1e-1,
    init="randn",
    filter="none",
    lr_decay=True,
    scoring_choice="loss",
    BN=0.001,
    Group=0.01,
)


def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f"Deprecated key in config dict: {key}!")
    return config


"""
-AGR setting
-AGR condition (whether the targeted sample in the aggregation)
-GIA method
-Inversion Framework
"""


def mkrum_setting(fl_setting):
    batch_size = fl_setting["batch_size"]
    client_num = fl_setting["client_num"]
    malicious_num = fl_setting["malicious_num"]

    drop_num = 2 * malicious_num + 2
    agg_num = client_num - drop_num
    left_bengin_num = agg_num - malicious_num
    without_target_benign_num = left_bengin_num - 1
    with_target_benign_num = drop_num + 1
    without_target_mal_num = malicious_num

    agr_setting = {
        "without_target_benign_num": without_target_benign_num,
        "with_target_benign_num": with_target_benign_num,
        "without_target_mal_num": without_target_mal_num,
    }

    return agr_setting


def bulyan_setting(fl_setting):
    batch_size = fl_setting["batch_size"]
    client_num = fl_setting["client_num"]
    malicious_num = fl_setting["malicious_num"]

    drop_num = 4 * malicious_num
    agg_num = client_num - drop_num
    left_bengin_num = agg_num - malicious_num
    without_target_benign_num = left_bengin_num - 1
    with_target_benign_num = drop_num + 1
    without_target_mal_num = malicious_num

    agr_setting = {
        "without_target_benign_num": without_target_benign_num,
        "with_target_benign_num": with_target_benign_num,
        "without_target_mal_num": without_target_mal_num,
    }

    return agr_setting


def afa_setting(fl_setting):
    batch_size = fl_setting["batch_size"]
    client_num = fl_setting["client_num"]
    malicious_num = fl_setting["malicious_num"]

    without_target_benign_num = client_num - malicious_num - 1
    with_target_benign_num = 1
    without_target_mal_num = malicious_num

    agr_setting = {
        "without_target_benign_num": without_target_benign_num,
        "with_target_benign_num": with_target_benign_num,
        "without_target_mal_num": without_target_mal_num,
    }

    return agr_setting


def fang_setting(fl_setting):
    batch_size = fl_setting["batch_size"]
    client_num = fl_setting["client_num"]
    malicious_num = fl_setting["malicious_num"]

    without_target_benign_num = client_num - malicious_num - 1
    with_target_benign_num = 1
    without_target_mal_num = malicious_num

    agr_setting = {
        "without_target_benign_num": without_target_benign_num,
        "with_target_benign_num": with_target_benign_num,
        "without_target_mal_num": without_target_mal_num,
    }

    return agr_setting


def standard_setting(fl_setting):
    batch_size = fl_setting["batch_size"]
    client_num = fl_setting["client_num"]
    malicious_num = fl_setting["malicious_num"]

    without_target_benign_num = client_num - malicious_num - 1
    with_target_benign_num = 1
    without_target_mal_num = malicious_num

    agr_setting = {
        "without_target_benign_num": without_target_benign_num,
        "with_target_benign_num": with_target_benign_num,
        "without_target_mal_num": without_target_mal_num,
    }

    return agr_setting


def sp_setting(fl_setting):
    batch_size = fl_setting["batch_size"]
    client_num = fl_setting["client_num"]
    malicious_num = fl_setting["malicious_num"]
    target_number = fl_setting["target_number"]

    without_target_benign_num = client_num - malicious_num - target_number
    with_target_benign_num = target_number
    without_target_mal_num = malicious_num

    agr_setting_dict = {
        "without_target_benign_num": without_target_benign_num,
        "with_target_benign_num": with_target_benign_num,
        "without_target_mal_num": without_target_mal_num,
    }

    return agr_setting_dict


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        use_ddim=False,
        model_path=str(IMAGENET_DIFFUSION_CHECKPOINT),
        classifier_path=str(IMAGENET_CLASSIFIER_CHECKPOINT),
        classifier_scale=1,  ##1.0
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def evaluate1(labels, output_size):
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    args.batch_size = output_size
    args.num_samples = output_size
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        # classes = torch.randint(
        #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        # )
        classes = labels
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [
            torch.zeros_like(sample) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [
            torch.zeros_like(classes) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    images = []
    index = []
    for i in range(args.batch_size):
        index.append(str(i))
    print(index)
    for i in range(args.batch_size):
        data = arr[i, :]  # i为第几个数据
        print(np.shape(data))

        image_tensor_from_numpy2 = cv2.cvtColor(
            data, cv2.COLOR_BGR2RGB
        )  # 先将 BGR 变为 RGB
        image_tensor_from_numpy2 = torch.from_numpy(
            image_tensor_from_numpy2.transpose((2, 0, 1))
        )
        image_tensor_from_numpy2 = image_tensor_from_numpy2.float().div(
            255
        )  # 归一化到[0,1]
        images.append(image_tensor_from_numpy2)
        torchvision.utils.save_image(
            image_tensor_from_numpy2,
            os.path.join(str(IMAGES_DIR), index[i] + f"imagenet.png"),
        )
    images = torch.stack(images, 0)
    torchvision.utils.save_image(
        images, os.path.join(str(IMAGES_DIR), f"16_imagenet.png")
    )
    print(images.shape)
    trueimage = []
    for i in range(args.batch_size):
        img_path = os.path.join(str(IMAGES_DIR), index[i] + f"imagenet.png")
        transform1 = transforms.Compose(
            [
                transforms.CenterCrop((112, 112)),  # 只能对PIL图片进行裁剪
                transforms.ToTensor(),
            ]
        )
        img_PIL = Image.open(img_path).convert("RGB")
        img_PIL_Tensor = transform1(img_PIL)
        # img_PIL_Tensor = img_PIL_Tensor.permute(1, 2, 0)
        print(type(img_PIL))
        print(type(img_PIL_Tensor))
        print(img_PIL_Tensor.shape)
        trueimage.append(img_PIL_Tensor)
    trueimage = torch.stack(trueimage, 0)
    print(trueimage.shape)
    torchvision.utils.save_image(
        trueimage, os.path.join(str(IMAGES_DIR), f"trueb8_imagenet.png")
    )
    return trueimage.to(device)


def mkrum_agr_condition(candidate_indices, input, with_target_benign_num):
    target_img_batch = []
    target_batch_index = None
    flag = False
    if np.sum(np.array(candidate_indices) < with_target_benign_num) == 1:
        target_client_index = np.array(candidate_indices)[
            np.array(candidate_indices) < with_target_benign_num
        ]
        target_img_batch = input[target_client_index[0]]
        target_batch_index = target_client_index[0]
        flag = True
    return flag, target_img_batch, target_batch_index


def bulyan_agr_condition(candidate_indices, input, with_target_benign_num):
    target_img_batch = []
    target_batch_index = None
    flag = False
    if np.sum(np.array(candidate_indices) < with_target_benign_num) == 1:
        target_client_index = np.array(candidate_indices)[
            np.array(candidate_indices) < with_target_benign_num
        ]
        target_img_batch = input[target_client_index[0]]
        target_batch_index = target_client_index[0]
        flag = True
    return flag, target_img_batch, target_batch_index


def afa_agr_condition(candidate_indices, input, with_target_benign_num):
    target_img_batch = []
    flag = False
    if 0 in candidate_indices:
        target_img_batch = input[0]
        flag = True
    return flag, target_img_batch, 0


def fang_agr_condition(candidate_indices, input, with_target_benign_num):
    target_img_batch = []
    flag = False
    if 0 in candidate_indices:
        target_img_batch = input[0]
        flag = True
    return flag, target_img_batch, 0


def standard_agr_condition(candidate_indices, input, with_target_benign_num):
    target_img_batch = []
    flag = False
    if 0 in candidate_indices:
        target_img_batch = input[0]
        flag = True
    return flag, target_img_batch, 0


def sp_agr_condition(candidate_indices, input, with_target_benign_num):
    target_img_batch = []
    target_number = with_target_benign_num
    flag = False
    target_indices = list(range(target_number))
    hit_num = 0
    target_batch_index = None
    for target_index in target_indices:
        if target_index in candidate_indices:
            target_img_batch = input[target_index]
            target_batch_index = target_index
            hit_num += 1

    if hit_num == 1:
        flag = True

    return flag, target_img_batch, target_batch_index


"""
Inversion Attack

"""

# Total Variation Regularization


def total_variation(x):

    diff1 = torch.norm(x[:, :, :, :-1] - x[:, :, :, 1:])
    diff2 = torch.norm(x[:, :, :-1, :] - x[:, :, 1:, :])
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy
    # return diff1+diff2


# L2 Regularization


def l2_norm(x):

    return torch.norm(x, 2)


# Clip Regularization


def clip_loss(x):
    clip_x = x.detach().clone()
    clip_x.data[clip_x.data < 0] = 0.0
    clip_x.data[clip_x.data > 1] = 1.0
    return torch.norm(x - clip_x, 2)


# Scale Regularization


def scale_loss(x):
    scale_x = x.detach().clone()
    scale_x.data = (scale_x.data - torch.min(scale_x.data)) / (
        torch.max(scale_x.data) - torch.min(scale_x.data)
    )
    return torch.norm(x - scale_x, 2)


# Initlization


def init_dummy(shape, method="u"):
    if method == "u":
        dummy_data = torch.distributions.uniform.Uniform(
            torch.tensor([0.0]), torch.tensor([1.0])
        ).sample(shape)
        return dummy_data.to(device).requires_grad_(True)
    else:
        dummy_data = torch.randn(shape).to(device).requires_grad_(True)
        return dummy_data.to(device).requires_grad_(True)


# Core Attack Method


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def _gradient_closure(
    model,
    x_group,
    optimizer,
    dL_dW,
    dummy_data,
    dummy_label,
    l2_coeff=1e-6,
    tv_coeff=1e-4,
    group_coeff=1e-4,
    clip_coeff=1.0,
    scale_coeff=0.1,
):
    config = DEFAULT_CONFIG

    def closure():
        optimizer.zero_grad()
        model.zero_grad()
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fn(model(dummy_data), dummy_label)
        gradient = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        loss_grad = reconstruction_costs(
            [gradient],
            dL_dW,
            cost_fn=config["cost_fn"],
            indices=config["indices"],
            weights=config["weights"],
        )
        bn_loss = 0
        loss_tv = total_variation(dummy_data)
        loss_l2 = l2_norm(dummy_data)
        loss_group = 0
        if group_coeff > 0 and x_group != None:
            # group_loss = ((x_trial - self.x_group) ** 2).mean()
            # alh = (x_trial - self.x_group) / (self.x_max - self.x_group)
            alh = 1.8  ##1.8
            # group_loss = ((x_trial - (self.x_group * alh)) ** 2).mean()
            # group_loss = ((x_trial - ((self.x_group + self.x_max + x_trial) / 3 * alh)) ** 2).mean()
            loss_group = ((dummy_data - ((x_group + dummy_data) / 2 * alh)) ** 2).mean()
            # group_loss = ((x_trial - ((0.7 * self.x_group + 0.8 * x_trial) * alh)) ** 2).mean()

        loss_clip = clip_loss(dummy_data)
        loss_scale = scale_loss(dummy_data)

        loss_aux = (
            group_coeff * loss_group
            + l2_coeff * loss_l2
            + tv_coeff * loss_tv
            + clip_coeff * loss_clip
            + scale_coeff * loss_scale
        )
        rec_loss = loss_grad + loss_aux
        rec_loss.backward()
        if config["signed"]:
            dummy_data.grad.sign_()
        return (
            loss_grad,
            rec_loss,
            bn_loss,
            loss_group,
            loss_tv,
            loss_clip,
            loss_scale,
            loss_aux,
            loss_l2,
        )

    return closure


def gardient_inversion_attack(
    x_group,
    model,
    gradient,
    target_class,
    original_input,
    save_name,
    logger,
    image_shape,
    output_size=1,  # Output Image Number
    epoch_num=20000,
    grad_coeff=1e-1,
    l2_coeff=1e-6,
    tv_coeff=1e-4,
    group_coeff=1e-4,
    clip_coeff=1.0,
    scale_coeff=0.1,
    lr=0.1,
    save_path="./result/",
    only_last=True,
    ifprint=False,
):

    vutils.save_image(
        original_input,
        "./output/" + save_name + "_Original.png",
        normalize=False,
        scale_each=True,
        nrow=int(4),
    )

    criterion = nn.CrossEntropyLoss()
    shape = [output_size, *image_shape]

    random.seed(0)
    torch.manual_seed(0)
    model.zero_grad()

    # Cal original graidnet
    model.eval()
    dL_dW = gradient

    # Create dummy data
    dummy_label = torch.LongTensor([target_class] * output_size).to(device)
    dummy_data = torch.randn(shape).to(device).requires_grad_(True)
    # dummy_data = evaluate1(dummy_label, output_size)
    original_input = original_input.detach().clone().to(device)
    # Set optimizer
    optimizer = torch.optim.Adam([dummy_data], lr=lr)  # ,betas=[0.5, 0.9], eps = 1e-8)
    # lr_scheduler = lr_cosine_policy(lr, 50, epoch_num)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[epoch_num // 2.667, epoch_num // 1.6, epoch_num // 1.142],
        gamma=0.1,
    )  # 3/8 5/8 7/8

    history = []
    final_loss = 0
    try:
        for t in range(epoch_num):
            closure = _gradient_closure(
                model,
                x_group,
                optimizer,
                dL_dW,
                dummy_data,
                dummy_label,
                l2_coeff=0,
                tv_coeff=0.15,
                group_coeff=1e-4,
                clip_coeff=0,
                scale_coeff=0,
            )
            (
                loss_grad,
                rec_loss,
                bn_loss,
                loss_group,
                loss_tv,
                loss_clip,
                loss_scale,
                loss_aux,
                loss_l2,
            ) = optimizer.step(closure)
            lr_scheduler.step()
            final_loss = rec_loss
            if (t + 1) % 1000 == 0:
                if ifprint:
                    print("------------------Current Iter %d----------------" % t)
                    print("Grad Loss: %.4f" % loss_grad)
                    print("TV Loss: %.4f" % loss_tv)
                    print("L2 Loss: %.4f" % loss_l2)
                    print("Group Loss: %.4f" % loss_group)
                    print("Clip Loss: %.4f" % loss_clip)
                    print("Scale Loss: %.4f" % loss_scale)
                    print("AUX Loss: %.4f" % loss_aux)
                    print("Total Loss: %.4f" % rec_loss)
                    print("#######Eval#######")
                    for i in range(output_size):
                        print("-----Target %d-----" % i)
                        print(
                            "PSNR: %.8f"
                            % cal_PSNR(dummy_data[i : i + 1], original_input[i : i + 1])
                        )
                        print(
                            "LPIPS: %.8f"
                            % cal_LPIPS(
                                dummy_data[i : i + 1], original_input[i : i + 1]
                            )
                        )
                        print(
                            "SSIM: %.8f"
                            % cal_SSIM(dummy_data[i : i + 1], original_input[i : i + 1])
                        )
                if logger != None:
                    logger.info("------------------Current Iter %d----------------" % t)
                    logger.info("Grad Loss: %.4f" % loss_grad)
                    logger.info("TV Loss: %.4f" % loss_tv)
                    logger.info("L2 Loss: %.4f" % loss_l2)
                    logger.info("Group Loss: %.4f" % loss_group)
                    logger.info("Clip Loss: %.4f" % loss_clip)
                    logger.info("Scale Loss: %.4f" % loss_scale)
                    logger.info("AUX Loss: %.4f" % loss_aux)
                    logger.info("Total Loss: %.4f" % rec_loss)
                    logger.info("#######Eval#######")
                    for i in range(output_size):
                        logger.info("-----Target %d-----" % i)
                        logger.info(
                            "PSNR: %.8f"
                            % cal_PSNR(dummy_data[i : i + 1], original_input[i : i + 1])
                        )
                        logger.info(
                            "LPIPS: %.8f"
                            % cal_LPIPS(
                                dummy_data[i : i + 1], original_input[i : i + 1]
                            )
                        )
                        logger.info(
                            "SSIM: %.8f"
                            % cal_SSIM(dummy_data[i : i + 1], original_input[i : i + 1])
                        )

                if not only_last:
                    history.append(dummy_data.cpu())

                vutils.save_image(
                    dummy_data.reshape([*shape]),
                    "./output/" + save_name + "_{}.png".format(t),
                    normalize=False,
                    scale_each=True,
                    nrow=int(4),
                )

        if only_last:
            history.append(dummy_data.cpu())

    except KeyboardInterrupt:
        print("Stopping: the iter is %d" % t)

    return history, history[-1], final_loss


"""
FedSGD Gradient Inference while Framework Running
"""


def fedSGD_replacement_batch_gradient_inference_under_AGR(
    server,
    clients,
    input,
    replacement_state_dict,
    previous_aggregated_gradient,
    runtime_logger,
    coeff=[1, 0, 0, 0],
    poi_index=[0, 1],
    return_candidx=True,
    scale=100,
    **kwargs,
):

    client_num = len(clients)

    gradient_list = []
    bn_data_list = []

    # if robust_agg == None:
    #     robust_agg = server.aggregate_grads

    for client in clients:
        client.set_server(server)

    last_global_weights = server.get_weights()
    previous_global_model_state_dict = copy.deepcopy(server.global_model.state_dict())

    malicious_gradient_flag = False
    ref_benign_gradient_list = []

    for index, train_tuple in enumerate(zip(clients, input)):
        if poi_index[index] == 1:
            client, input_tuple = train_tuple
            client.client_model.eval()

            X, y = input_tuple

            ref_benign_gradient = client.cal_gradient(X, y)
            ref_benign_gradient_list.append(ref_benign_gradient)

    average_ref_benign_gradient = cal_gradient_mean(ref_benign_gradient_list)

    for index, train_tuple in enumerate(zip(clients, input)):
        client, input_tuple = train_tuple
        client.client_model.eval()

        X, y = input_tuple

        if poi_index[index] == 1:
            if not malicious_gradient_flag:
                malicious_gradient_flag = True

                gradient_poi = calculate_replacement_gradient(
                    client.client_model, replacement_state_dict, scale=scale
                )

                gradient_AGR = previous_aggregated_gradient
                last_gradient_AGR = copy.deepcopy(gradient_AGR)

                gradient_m = optimize_malicious_update(
                    optimize_mal_by_unit_scale_with_ref,
                    gradient_AGR,
                    gradient_poi,
                    average_ref_benign_gradient,
                    runtime_logger,
                    coeff=coeff,
                    ifprint=False,
                    **kwargs,
                )
                gradient_malicious = gradient_m
                gradient = gradient_malicious
            else:
                gradient = gradient_malicious

            accuracy, loss_value = client.test_batch(X, y)
            bn_data = client.get_bn()

        else:
            gradient, bn_data, accuracy, loss_value = client.train_sgd_batch(X, y)

        gradient_list.append(gradient)
        bn_data_list.append(bn_data)

    aggregation, candidate_indices = server.robust_aggregate_grads(
        gradient_list, return_candidx=True
    )
    server.train_sgd_batch(gradient_list, bn_data_list)

    current_global_weights = server.get_weights()

    inference_gradient = infer_global_aggregated_gradient(
        last_global_weights, current_global_weights, scale=scale
    )

    lr = server.get_lr()
    print("#Client:", client_num)
    print("Lr:", lr)

    if return_candidx:
        return (
            gradient_list,
            inference_gradient,
            previous_global_model_state_dict,
            candidate_indices,
        )
    else:
        return gradient_list, inference_gradient, previous_global_model_state_dict


"""
Fast Method
"""


def get_agr(
    agr_name, server, dataset_name, test_dataset_dict, agr_logger, ifprint=False
):
    agr_method_dict = {
        "MKrum": aggregate_robust_multi_krum,
        "AFA": aggregate_robust_FL_trust,
        "Bulyan": aggregate_robust_bulyan,
        "Fang": aggregate_robust_fang,
        "Standard": aggregate_standard,
    }

    class_dict = {
        "TinyImageNet": 200,
        "Cifar100": 100,
        "CalTech256": 256,
        "MyImageNet": 100,
    }

    val_require_dict = {
        "MKrum": False,
        "AFA": True,
        "Bulyan": False,
        "Fang": True,
        "Standard": False,
    }

    if val_require_dict[agr_name]:
        val_class_index = random.sample(
            list(range(0, class_dict[dataset_name])),
            k=int(class_dict[dataset_name] * 0.6),
        )
        # val_class_index = random.sample(list(range(0,class_dict[dataset_name])),k=20)
        val_class_index.sort()
        print("Validation Dataset Class index:", val_class_index)
        val_data = sampling_dataset(
            test_data_dict[dataset_name],
            label_range=class_dict[dataset_name],
            each_class_num=5,
            label_index=val_class_index,
        )
        return agr_method_dict[agr_name](server, val_data, agr_logger, ifprint=False)
    else:
        return agr_method_dict[agr_name](agr_logger, ifprint=False)


def gradient_simulate(
    dataset_name,
    dataset_info,
    target_class,
    knowledge_level,
    agr_name,
    fl_setting,
    agr_attack_setting_dict,
    sp=False,
    model_dataset=None,
):
    # Config
    batch_size = fl_setting["batch_size"]
    client_num = fl_setting["client_num"]
    malicious_num = fl_setting["malicious_num"]

    agr_setting_dict = {
        "MKrum": mkrum_setting,
        "AFA": afa_setting,
        "Bulyan": bulyan_setting,
        "Fang": fang_setting,
        "Standard": standard_setting,
    }

    agr_condition_dict = {
        "MKrum": mkrum_agr_condition,
        "AFA": afa_agr_condition,
        "Bulyan": bulyan_agr_condition,
        "Fang": fang_agr_condition,
        "Standard": standard_agr_condition,
    }

    data_dict = dataset_info["train_dataset_dict"]
    test_data_dict = dataset_info["test_dataset_dict"]
    dataset_loader_dict = dataset_info["train_loader_dict"]
    dataset_test_loader_dict = dataset_info["test_loader_dict"]

    model_generator_dict = {
        "TinyImageNet": TinyImageNet_model_generator,
        "Cifar100": Cifar100_model_generator,
        "CalTech256": CalTech256_model_generator,
        "MyImageNet": MyImageNet_model_generator,
    }

    agr_setting = agr_setting_dict[agr_name](fl_setting)
    if sp:
        agr_setting = sp_setting(fl_setting)

    without_target_benign_num = agr_setting["without_target_benign_num"]
    with_target_benign_num = agr_setting["with_target_benign_num"]
    without_target_mal_num = agr_setting["without_target_mal_num"]

    poi_index = [0] * (client_num - malicious_num) + [1] * malicious_num

    server = Server(model_generator_dict[dataset_name], optim.SGD, {"lr": 0.01})
    clients = [
        Client(model_generator_dict[dataset_name], optim.SGD, {"lr": 0.01})
        for _ in range(client_num)
    ]

    model_path = (
        "./model/"
        + dataset_name
        + "_"
        + knowledge_level
        + "_"
        + str(target_class)
        + "_"
        + agr_name
        + "_Poisoning_Model.pth"
    )
    server.load_model(model_path)

    for index, client in enumerate(clients):
        client.load_model(model_path)

    targetGenerator = TargetBatchGenerator(
        data_dict[dataset_name], target_class, batch_size
    )
    target_train_loaders = [targetGenerator for _ in range(with_target_benign_num)]
    no_target_datasets = sampling_no_target_class_datasets(
        data_dict[dataset_name],
        without_target_mal_num + without_target_benign_num,
        target_class,
        dcopy=False,
    )

    train_loaders = [target_loader for target_loader in target_train_loaders]
    train_loaders += [
        iter(
            torch.utils.data.DataLoader(
                no_target_dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(0),
            )
        )
        for no_target_dataset in no_target_datasets
    ]

    target_model_path = (
        "./model/"
        + dataset_name
        + "_"
        + knowledge_level
        + "_"
        + str(target_class)
        + "_Malicious_Model.pth"
    )
    replacement_state_dict = torch.load(target_model_path)

    agr_logger = get_logger(
        "./log/S3/"
        + dataset_name
        + "_"
        + knowledge_level
        + "_"
        + str(target_class)
        + "_"
        + agr_name
        + "_Gradient.log",
        verbosity=1,
        name=dataset_name
        + "_"
        + knowledge_level
        + "_"
        + str(target_class)
        + "_"
        + agr_name
        + "_Gradient",
    )
    agr = get_agr(
        agr_name, server, dataset_name, test_data_dict, agr_logger, ifprint=False
    )

    server.set_robust_agg(
        agr,
        malicious_num,
        mal_client_idx=list(range(client_num - malicious_num, client_num)),
    )

    target_img_list = []
    agr_condition = agr_condition_dict[agr_name]
    if sp:
        agr_condition = sp_agr_condition

    for e in range(100):
        input = [next(loader) for loader in train_loaders]
        gradient_list, inference_gradient, previous_global_model_state_dict = (
            fedSGD_batch_gradient_inference_under_AGR(
                server, clients, input, return_candidx=False
            )
        )
        (
            gradient_list,
            inference_gradient,
            previous_global_model_state_dict,
            candidate_indices,
        ) = fedSGD_replacement_batch_gradient_inference_under_AGR(
            server,
            clients,
            input,
            replacement_state_dict,
            inference_gradient,
            agr_logger,
            coeff=[1, 0, 0, 0],
            poi_index=poi_index,
            return_candidx=True,
            scale=100,
            **agr_attack_setting_dict,
        )

        candidate_indices.sort()
        print("Without target client num:", without_target_benign_num)
        print(candidate_indices)
        flag, target_img_batch, target_batch_index = agr_condition(
            candidate_indices, input, with_target_benign_num
        )
        if flag:
            print("Hit, end in %d" % e)

            model = model_generator_dict[dataset_name]()
            model.load_state_dict(previous_global_model_state_dict)
            candidate_input = []
            select_index = -1
            for candi_index in candidate_indices:
                if candi_index == target_batch_index:
                    select_index = len(candidate_input)
                    # print(select_index)
                candidate_input.append(input[candi_index])
            print(len(candidate_input))
            batch_target_loss, batch_other_loss, batch_total_loss, batch_loss_gap = (
                cal_loss_gap_based_on_batch(
                    candidate_input,
                    model,
                    target_batch_index=[select_index],
                    target_index=[0],
                    ifprint=True,
                )
            )
            break

    return (
        target_img_batch,
        gradient_list,
        inference_gradient,
        previous_global_model_state_dict,
    )


def inversion(
    x_group,
    dataset_name,
    target_class,
    knowledge_level,
    agr_name,
    gradient_tuple,
    inversion_args=[1e-1, 1e-2, 10, 0.001],
    **kwargs,
):

    model_generator_dict = {
        "TinyImageNet": TinyImageNet_model_generator,
        "Cifar100": Cifar100_model_generator,
        "CalTech256": CalTech256_model_generator,
        "MyImageNet": MyImageNet_model_generator,
    }

    image_shape_dict = {
        "TinyImageNet": [1, 3, 64, 64],
        "Cifar100": [1, 3, 32, 32],
        "CalTech256": [1, 3, 112, 112],
        "MyImageNet": [1, 3, 112, 112],
    }

    (
        target_img_batch,
        gradient_list,
        inference_gradient,
        previous_global_model_state_dict,
    ) = gradient_tuple

    model = model_generator_dict[dataset_name]()
    model.load_state_dict(previous_global_model_state_dict)

    X, y = target_img_batch
    original_image = X[0:1]
    vutils.save_image(
        original_image.reshape(*image_shape_dict[dataset_name]),
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
    history, final_inversion_image, final_loss = gardient_inversion_attack(
        x_group,
        model,
        inference_gradient,
        target_class,
        original_image,
        save_name,
        inversion_logger,
        image_shape=image_shape_dict[dataset_name][1:],
        epoch_num=20000,
        lr=inversion_args[0],
        tv_coeff=inversion_args[1],
        grad_coeff=inversion_args[2],
        group_coeff=inversion_args[3],
        **kwargs,
    )

    final_inversion_image = final_inversion_image.to(device)
    vutils.save_image(
        final_inversion_image.reshape(*image_shape_dict[dataset_name]),
        "./result/"
        + dataset_name
        + "_"
        + knowledge_level
        + "_"
        + str(target_class)
        + "_"
        + agr_name
        + "_Inversion.png",
        normalize=False,
        scale_each=True,
        nrow=int(4),
    )

    eval_image(final_inversion_image, original_image)
    tp = transforms.ToPILImage()
    plt.figure(figsize=(10, 5))
    plt.suptitle(dataset_name + "_" + agr_name + "_" + knowledge_level)
    plt.subplot(1, 2, 1),
    plt.imshow(tp(original_image[0])), plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(tp(final_inversion_image[0])), plt.axis("off")
    plt.show()


def once_inversion(
    dataset_name,
    target_class,
    knowledge_level,
    agr_name,
    fl_setting,
    agr_attack_setting_dict,
    sp,
    **kwargs,
):
    gradient_tuple = gradient_simulate(
        dataset_name,
        target_class,
        knowledge_level,
        agr_name,
        fl_setting,
        agr_attack_setting_dict,
        sp,
    )
    inversion(
        dataset_name, target_class, knowledge_level, agr_name, gradient_tuple, **kwargs
    )


def inversion_fast_replay(setting_list):
    try:
        for setting in setting_list:
            (
                fl_setting,
                target_class,
                dataset_name,
                knowledge_level,
                agr_name,
                agr_attack_setting_dict,
                sp,
            ) = setting
            once_inversion(
                dataset_name,
                target_class,
                knowledge_level,
                agr_name,
                fl_setting,
                agr_attack_setting_dict,
                sp,
            )
    except KeyboardInterrupt:
        print("Stopping")


def reconstruction_costs(
    gradients, input_gradient, cost_fn="l2", indices="def", weights="equal"
):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == "def":
        indices = torch.arange(len(input_gradient))
    elif indices == "batch":
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == "topk-1":
        _, indices = torch.topk(
            torch.stack([p.norm() for p in input_gradient], dim=0), 4
        )
    elif indices == "top10":
        _, indices = torch.topk(
            torch.stack([p.norm() for p in input_gradient], dim=0), 10
        )
    elif indices == "top50":
        _, indices = torch.topk(
            torch.stack([p.norm() for p in input_gradient], dim=0), 50
        )
    elif indices in ["first", "first4"]:
        indices = torch.arange(0, 4)
    elif indices == "first5":
        indices = torch.arange(0, 5)
    elif indices == "first10":
        indices = torch.arange(0, 10)
    elif indices == "first50":
        indices = torch.arange(0, 50)
    elif indices == "last5":
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == "last10":
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == "last50":
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == "linear":
        weights = torch.arange(
            len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device
        ) / len(input_gradient)
    elif weights == "exp":
        weights = torch.arange(
            len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device
        )
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == "topk-2":
            _, indices = torch.topk(
                torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4
            )
        for i in indices:
            if cost_fn == "l2":
                costs += (
                    (trial_gradient[i] - input_gradient[i]).pow(2)
                ).sum() * weights[i]
            elif cost_fn == "l1":
                costs += (
                    (trial_gradient[i] - input_gradient[i]).abs()
                ).sum() * weights[i]
            elif cost_fn == "max":
                costs += (
                    (trial_gradient[i] - input_gradient[i]).abs()
                ).max() * weights[i]
            elif cost_fn == "sim":
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == "simlocal":
                costs += (
                    1
                    - torch.nn.functional.cosine_similarity(
                        trial_gradient[i].flatten(),
                        input_gradient[i].flatten(),
                        0,
                        1e-10,
                    )
                    * weights[i]
                )
        if cost_fn == "sim":
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)
