"""Mechanisms for image reconstruction from parameter gradients."""
import argparse
from cggl_paths import (
    CIFAR_DIFFUSION_CHECKPOINT,
    IMAGENET_DIFFUSION_CHECKPOINT,
    IMAGENET_CLASSIFIER_CHECKPOINT,
    IMAGES_DIR,
)
import os
from PIL import Image
import cv2
import numpy as np
import torch
from collections import defaultdict, OrderedDict
import torch.nn.functional as F
import torchvision
import inversefed
from torchvision import transforms
from inversefed.nn import MetaMonkey
import torch.distributed as dist
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
from skimage.metrics import structural_similarity as ssim
from torchvision import utils
from lpips import lpips
from .metrics import total_variation as TV
from .metrics import InceptionScore
from .medianfilt import MedianPool2d
from copy import deepcopy
from tqdm import trange


from simpleAICV.diffusion_model.models.diffusion_unet import DiffusionUNet
from simpleAICV.diffusion_model.diffusion_methods import DDPMSampler
from simpleAICV.diffusion_model.datasets.cifar100dataset import CIFAR100Dataset
from simpleAICV.diffusion_model.common import Opencv2PIL, TorchMeanStdNormalize, ClassificationCollater
from simpleAICV.classification.common import load_state_dict
from simpleAICV.diffusion_model.metrics.inception import InceptionV3

import torch
import torchvision.transforms as transforms



import time

DEFAULT_CONFIG = dict(signed=False,
                      boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-1,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss',
                      BN=0.001,
                      Group=0.01,
                      Clip=0,
                      Scale=0
                      )

device = torch.device('cuda:0')


def dfcifar(labels):
    num_classes = 100
    input_image_size = 32
    time_step = 1000
    batch_size = 8

    model = DiffusionUNet(inplanes=3,
                          planes=128,
                          planes_multi=[1, 2, 2, 2],
                          time_embedding_ratio=4,
                          block_nums=2,
                          dropout_prob=0,
                          num_groups=32,
                          use_attention_planes_multi_idx=[1],
                          num_classes=num_classes,
                          use_gradient_checkpoint=False)
    model.eval()
    sampler = DDPMSampler(beta_schedule_mode='linear',
                          linear_beta_1=1e-4,
                          linear_beta_t=0.02,
                          cosine_s=0.008,
                          t=1000,
                          mean_type='epsilon',
                          var_type='fixedsmall',
                          clip_denoised=True)

    # load pretrained model or not
    trained_model_path = str(CIFAR_DIFFUSION_CHECKPOINT)
    load_state_dict(trained_model_path, model)
    # labels = np.array([0., 1., 2., 3.]).astype(np.float32)
    # labels = torch.from_numpy(labels).long()
    all_step_images, last_step_images = sampler(model, [batch_size, 3, 32, 32],
                                                     class_label=labels.cpu(),
                                                     return_intermediates=True)
    index = []
    for i in range(batch_size):
        index.append(str(i))
    for i in range(batch_size):
        data = last_step_images[i, :]  # i为第几个数据
        print(np.shape(data))
        #data = np.uint8(data)
        #data = data.transpose(1, 2, 0)
        #image_tensor_from_numpy2 = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)  # 先将 BGR 变为 RGB
        image_tensor_from_numpy2 = torch.from_numpy(data)  # 先将 BGR 变为 RGB) #data.transpose((2, 0, 1))
        #image_tensor_from_numpy2 = image_tensor_from_numpy2.float().div(255)  # 归一化到[0,1]
        #images.append(image_tensor_from_numpy2)
        torchvision.utils.save_image(image_tensor_from_numpy2,
                                     os.path.join(str(IMAGES_DIR),
                                                  index[i] + f"dfcifar.png"))


    trueimage=[]
    for i in range(batch_size):
        img_path = os.path.join(str(IMAGES_DIR),
                                                  index[i] + f"dfcifar.png")
        transform1 = transforms.Compose([
            transforms.CenterCrop((32, 32)),  # 只能对PIL图片进行裁剪
            transforms.ToTensor(),
        ]
        )
        img_PIL = Image.open(img_path).convert('RGB')
        img_PIL_Tensor = transform1(img_PIL)
        #img_PIL_Tensor = img_PIL_Tensor.permute(1, 2, 0)
        print(type(img_PIL))
        print(type(img_PIL_Tensor))
        print(img_PIL_Tensor.shape)
        trueimage.append(img_PIL_Tensor)
    trueimage = torch.stack(trueimage, 0)

    return trueimage.cuda()


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=8,
        batch_size=8,
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

def evaluate1(labels):
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

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

        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [torch.zeros_like(classes) for _ in range(dist.get_world_size())]
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

        image_tensor_from_numpy2 = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)  # 先将 BGR 变为 RGB
        image_tensor_from_numpy2 = torch.from_numpy(image_tensor_from_numpy2.transpose((2, 0, 1)))
        image_tensor_from_numpy2 = image_tensor_from_numpy2.float().div(255)  # 归一化到[0,1]
        images.append(image_tensor_from_numpy2)
        torchvision.utils.save_image(image_tensor_from_numpy2,
                                     os.path.join(str(IMAGES_DIR),
                                                  index[i] + f"imagenet.png"))
    images = torch.stack(images, 0)
    torchvision.utils.save_image(images,
                                 os.path.join(str(IMAGES_DIR), f"8_imagenet.png"))
    print(images.shape)
    trueimage = []
    for i in range(args.batch_size):
        img_path = os.path.join(str(IMAGES_DIR),
                                                  index[i] + f"imagenet.png")
        transform1 = transforms.Compose([
            transforms.CenterCrop((112, 112)),  # 只能对PIL图片进行裁剪
            transforms.ToTensor(),
        ]
        )
        img_PIL = Image.open(img_path).convert('RGB')
        img_PIL_Tensor = transform1(img_PIL)
        #img_PIL_Tensor = img_PIL_Tensor.permute(1, 2, 0)
        print(type(img_PIL))
        print(type(img_PIL_Tensor))
        print(img_PIL_Tensor.shape)
        trueimage.append(img_PIL_Tensor)
    trueimage = torch.stack(trueimage, 0)
    print(trueimage.shape)
    torchvision.utils.save_image(trueimage,
                                 os.path.join(str(IMAGES_DIR), f"trueb8_imagenet.png"))
    return trueimage.to(device)

def _label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config
# Clip Regularization

def clip_loss(x):
    clip_x = x.detach().clone()
    clip_x.data[clip_x.data < 0] = 0.
    clip_x.data[clip_x.data > 1] = 1.
    return torch.norm(x - clip_x, 2)

# Scale Regularization

def scale_loss(x):
    scale_x = x.detach().clone()
    scale_x.data = (scale_x.data - torch.min(scale_x.data)) / \
        (torch.max(scale_x.data) - torch.min(scale_x.data))
    return torch.norm(x - scale_x, 2)

class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1,
                 bn_loss_layers=None, x_group=None, x_max=None):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        self.mean_std = mean_std
        self.num_images = num_images
        self.bn_loss_layers = bn_loss_layers
        self.x_group = x_group
        self.x_max = x_max
        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.iDLG = True

    def reconstruct(self, ground_truth, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()

        stats = defaultdict(list)
        x = self._init_images(img_shape, labels)
        x_tv = x
        scores = torch.zeros(self.config['restarts'])

        if labels is None:
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                self.reconstruct_label = False
            else:
                # DLG label recovery
                # However this also improves conditioning for some LBFGS cases
                self.reconstruct_label = True

                def loss_fn(pred, labels):
                    labels = torch.nn.functional.softmax(labels, dim=-1)
                    return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))

                self.loss_fn = loss_fn
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False

        try:
            for trial in range(self.config['restarts']):
                x_trial, labels = self._run_trial(ground_truth, x, input_data, labels, x_tv, dryrun=dryrun)
                # x_trial, labels = self._run_trial(x[trial], input_data, labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x = x_trial
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            # x_optimal = x[optimal_index]
            x_optimal = x_trial
        print(f'Total time: {time.time() - start_time}.')
        return x_optimal.detach(), stats

    def _init_images(self, img_shape, labels):
        if self.config['init'] == 'randn':
            #return evaluate1(labels)
            return dfcifar(labels)
            #return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            # return evaluate(sampler, model1,
            # (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2).reshape(args.num_images, 3, 32, 32)
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            # return evaluate(sampler, model1,
            # torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)).reshape(args.num_images, 3, 32, 32)
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def _run_trial(self, ground_truth, x_trial, input_data, labels, x_tv, dryrun=False):
        #x_trial = x_trial.view(1, 3, 32, 32) ##

        x_trial = x_trial.squeeze() ##batch=1时注释掉
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.model(x_trial)
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial, labels])
            else:
                raise ValueError()
        else:
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial])
            else:
                raise ValueError()

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                         max_iterations // 1.142],
                                                             gamma=0.1)  # 3/8 5/8 7/8
        try:
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, ground_truth, x_trial, input_data, labels, x_tv)
                rec_loss, bn_loss, group_loss, TV_loss, Clip_loss, Scale_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.config['boxed']:
                        x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)

                    if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        elif self.config['filter'] == 'median':
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                        else:
                            raise ValueError()
                if (iteration + 1) % 1000 == 0:
                    #args = inversefed.options().parse_args()
                    #setup = inversefed.utils.system_startup(args)
                    #ground_truth = ground_truth.to(**setup)
                    y = ground_truth.squeeze(0)
                    lpips_loss = lpips.LPIPS(net='vgg', spatial=False).to(device)
                    lpips_score = 99.0
                    # score = lpips_loss(x_trial, y).squeeze().item()
                    # if score < lpips_score:
                    #     lpips_score = score
                    for i in range(8):
                        #y = ground_truth.squeeze(0)
                        score = lpips_loss(x_trial[i], y[i]).squeeze().item()
                        if score < lpips_score:
                            lpips_score = score

                    def calc_ssim(target, recon):
                        '''tructural similarity index'''
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
                        ssim_score = ssim(
                            target_np, recon_np,
                            data_range=255, multichannel=multichannel)

                        return ssim_score

                    output_denormalized1 = torch.clamp(x_trial * ds + dm, 0, 1)
                    ground_truth_denormalized1 = torch.clamp(y * ds + dm, 0, 1)
                    # ground_truth1 = (ground_truth-0.5).true_divide(0.5)
                    # ssim_score = calc_ssim(output, ground_truth)
                    ssim_score = calc_ssim(output_denormalized1, ground_truth_denormalized1)

                    test_mse = (x_trial - y).pow(2).mean().item()
                    #feat_mse = (model(x_trial) - model(input_data)).pow(2).mean().item()
                    test_psnr = inversefed.metrics.psnr(x_trial, y, factor=1 / ds)
                    print("-----Target %d-----" % iteration)
                    print("PSNR: %.8f" % test_psnr)
                    print("LPIPS: %.8f" % lpips_score)
                    print("SSIM: %.8f" % ssim_score)
                    print("MSE: %.8f" % test_mse)
                    print("TV Loss: %.4f" % TV_loss)
                    print("BN Loss: %.4f" % bn_loss)
                    print("group_loss: %.4f" % group_loss)
                    print("Clip_loss : %.4f" % Clip_loss)
                    print("Scale_loss : %.4f" % Scale_loss)
                    print("Total Loss: %.4f" % rec_loss)
                    print("#######Eval#######")
                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels

    def _gradient_closure(self, optimizer,ground_truth, x_trial, input_gradient, label,x_tv):

        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])
            bn_loss = 0
            group_loss = 0
            TV_loss = TV(x_trial)

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial+x_tv) * 0.5
            if self.config['Clip'] > 0:
                rec_loss += self.config['Clip'] * clip_loss(x_trial)
            if self.config['Scale'] > 0:
                rec_loss += self.config['Scale'] * scale_loss(x_trial)
            if self.config['BN'] > 0:
                bn_loss = sum([mod.bn_loss for mod in self.bn_loss_layers])
                rec_loss += self.config['BN'] * bn_loss
            if self.config['Group'] > 0 and self.x_group != None and self.x_max != None:
                # group_loss = ((x_trial - self.x_group) ** 2).mean()
                #alh = (x_trial - self.x_group) / (self.x_max - self.x_group)
                alh = 1.8 ##1.8
                # group_loss = ((x_trial - (self.x_group * alh)) ** 2).mean()
                # group_loss = ((x_trial - ((self.x_group + self.x_max + x_trial) / 3 * alh)) ** 2).mean()
                group_loss = ((x_trial - ((self.x_group + x_trial) / 2 * alh)) ** 2).mean()
                #group_loss = ((x_trial - ((0.7 * self.x_group + 0.8 * x_trial) * alh)) ** 2).mean()
                rec_loss += self.config['Group'] * group_loss
                # if (iteration + 1) % 500 == 0:
                #     print("------------------Current Iter %d----------------" % iteration)
                #     print("TV Loss: %.4f" % TV(x_trial))
                #     print("BN Loss: %.4f" % bn_loss)
                #     print("group_loss: %.4f" % group_loss)
                #     print("Total Loss: %.4f" % rec_loss)
                #     print("#######Eval#######")
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss, bn_loss, group_loss, TV_loss, clip_loss(x_trial), scale_loss(x_trial)

        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            return reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        if self.reconstruct_label:
            labels = self.model(x_optimal).softmax(dim=1)
        loss = self.loss_fn(self.model(x_optimal), labels)
        gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats


class FedAvgReconstructor(GradientReconstructor):
    """Reconstruct an image from weights after n gradient descent steps."""

    def __init__(self, model, mean_std=(0.0, 1.0), local_steps=2, local_lr=1e-4,
                 config=DEFAULT_CONFIG, num_images=1, use_updates=True, batch_size=0,
                 bn_loss_layers=None, ):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, mean_std, config, num_images)
        self.local_steps = local_steps
        self.local_lr = local_lr
        self.use_updates = use_updates
        self.batch_size = batch_size
        self.bn_loss_layers = bn_loss_layers

    def _gradient_closure(self, optimizer, ground_truth, x_trial, input_parameters, labels):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr,
                                    use_updates=self.use_updates,
                                    batch_size=self.batch_size)
            rec_loss = reconstruction_costs([parameters], input_parameters,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            # if self.config['total_variation'] > 0:
            #     rec_loss += self.config['total_variation'] * TV(x_trial)
            # if self.config['BN'] > 0:
            #     bn_loss = sum([mod.bn_loss for mod in self.bn_loss_layers])
            #     rec_loss += self.config['BN'] * bn_loss
            # rec_loss.backward()
            # if self.config['signed']:
            #     x_trial.grad.sign_()
            # return rec_loss
            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)
            if self.config['clip_loss'] > 0:
                rec_loss += self.config['clip_loss'] * clip_loss(x_trial)
            if self.config['scale_loss'] > 0:
                rec_loss += self.config['scale_loss'] * scale_loss(x_trial)
            if self.config['BN'] > 0:
                bn_loss = sum([mod.bn_loss for mod in self.bn_loss_layers])
                rec_loss += self.config['BN'] * bn_loss
            if self.config['Group'] > 0 and self.x_group != None and self.x_max != None:
                # group_loss = ((x_trial - self.x_group) ** 2).mean()
                #alh = (x_trial - self.x_group) / (self.x_max - self.x_group)
                alh = 1.8 ##1.8
                # group_loss = ((x_trial - (self.x_group * alh)) ** 2).mean()
                # group_loss = ((x_trial - ((self.x_group + self.x_max + x_trial) / 3 * alh)) ** 2).mean()
                group_loss = ((x_trial - ((self.x_group + x_trial) / 2 * alh)) ** 2).mean()
                #group_loss = ((x_trial - ((0.7 * self.x_group + 0.8 * x_trial) * alh)) ** 2).mean()
                rec_loss += self.config['Group'] * group_loss
                # if (iteration + 1) % 500 == 0:
                #     print("------------------Current Iter %d----------------" % iteration)
                #     print("TV Loss: %.4f" % TV(x_trial))
                #     print("BN Loss: %.4f" % bn_loss)
                #     print("group_loss: %.4f" % group_loss)
                #     print("Total Loss: %.4f" % rec_loss)
                #     print("#######Eval#######")
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss, bn_loss, group_loss, TV(x_trial), clip_loss(x_trial), scale_loss(x_trial)

        return closure

    def _score_trial(self, x_trial, input_parameters, labels):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr, use_updates=self.use_updates)
            return reconstruction_costs([parameters], input_parameters,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)


def loss_steps(model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4, use_updates=True,
               batch_size=0):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    if use_updates:
        patched_model_origin = deepcopy(patched_model)
    for i in range(local_steps):
        if batch_size == 0:
            outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        else:
            idx = i % (inputs.shape[0] // batch_size)
            outputs = patched_model(inputs[idx * batch_size:(idx + 1) * batch_size], patched_model.parameters)
            labels_ = labels[idx * batch_size:(idx + 1) * batch_size]
        loss = loss_fn(outputs, labels_).sum()
        grad = torch.autograd.grad(loss, patched_model.parameters.values(),
                                   retain_graph=True, create_graph=True, only_inputs=True)

        patched_model.parameters = OrderedDict((name, param - lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))

    if use_updates:
        patched_model.parameters = OrderedDict((name, param - param_origin)
                                               for ((name, param), (name_origin, param_origin))
                                               in zip(patched_model.parameters.items(),
                                                      patched_model_origin.parameters.items()))
    return list(patched_model.parameters.values())


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)
