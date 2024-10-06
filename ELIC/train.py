# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from ELICUtilis.datasets import ImageFolder

from tensorboardX import SummaryWriter
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
from ELICUtilis.utilis.utilis import DelfileList, load_checkpoint
from DWT_Network import TestModel
from Network import TestModel as Ori_TestModel


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, gate_weight=1, arch = "DWT"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.gate_weight = gate_weight
        self.arch = arch

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["y_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["y"]
        )
        out["z_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"]["z"]
        )
        out["mse_loss"] = self.mse(output["x_hat"], target) * 255 ** 2

        if "gate" in self.arch:
            # print(" loss gate ... \n")
            gate_activations = output["inp_atten"]
            # gate activation
            acts = torch.tensor([0.]).cuda()
            for ga in gate_activations:
                acts += torch.mean(ga)
            acts = torch.mean(acts / len(gate_activations))
            # print("acts loss", acts * self.gate_weight)
            # print("self.lmbda * out", self.lmbda * out["mse_loss"])
            # print("bpp_loss", out["bpp_loss"])

            out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"] + acts * self.gate_weight

        else:
            out["loss"] = self.lmbda * out["mse_loss"] + out["bpp_loss"]


        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate, betas=(0.9, 0.999),
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate, betas=(0.9, 0.999),
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, noisequant=True,
):
    model.train()
    device = next(model.parameters()).device
    train_loss = AverageMeter()
    train_bpp_loss = AverageMeter()
    train_y_bpp_loss = AverageMeter()
    train_z_bpp_loss = AverageMeter()
    train_mse_loss = AverageMeter()
    start = time.time()
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()


        out_net = model(d, noisequant)

        out_criterion = criterion(out_net, d)
        train_bpp_loss.update(out_criterion["bpp_loss"].item())
        train_y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
        train_z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
        train_loss.update(out_criterion["loss"].item())
        train_mse_loss.update(out_criterion["mse_loss"].item())

        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10000 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                f'\ty_Bpp loss: {out_criterion["y_bpp_loss"].item():.4f} |'
                f'\tz_Bpp loss: {out_criterion["z_bpp_loss"].item():.4f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
    print(f"Train epoch {epoch}: Average losses:"
          f"\tLoss: {train_loss.avg:.3f} |"
          f"\tMSE loss: {train_mse_loss.avg:.3f} |"
          f"\tBpp loss: {train_bpp_loss.avg:.4f} |"
          f"\ty_Bpp loss: {train_y_bpp_loss.avg:.5f} |"
          f"\tz_Bpp loss: {train_z_bpp_loss.avg:.5f} |"
          f"\tTime (s) : {time.time()-start:.4f} |"
          )


    return train_loss.avg, train_bpp_loss.avg, train_mse_loss.avg

def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    y_bpp_loss = AverageMeter()
    z_bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss().item())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            y_bpp_loss.update(out_criterion["y_bpp_loss"].item())
            z_bpp_loss.update(out_criterion["z_bpp_loss"].item())
            loss.update(out_criterion["loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\ty_Bpp loss: {y_bpp_loss.avg:.4f} |"
        f"\tz_Bpp loss: {z_bpp_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.4f}\n"
    )

    return loss.avg, bpp_loss.avg, mse_loss.avg


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def get_parser():
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "--N",
        default=192,
        type=int,
        help="Number of channels of main codec",
    )
    parser.add_argument(
        "--M",
        default=320,
        type=int,
        help="Number of channels of latent",
    )


##   基础层， 局部支路， 全局支路， 直接相加， 全局引导下的局部支路
    parser.add_argument(
        "--arch",
        default="base",
        type=str,
        help="arch [base, local, global, enhance, enhance_cab_local]",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        default=4000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=15e-3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--gate_weight",
        dest="gate_weight",
        type=float,
        default=1e-2,
        help="Channel wise gate weight",
    )

    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", default=1926, type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="use the pretrain model to refine the models",
    )
    parser.add_argument('--gpu-id', default='3', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--savepath', default='./checkpoint', type=str, help='Path to save the checkpoint')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    # args = parser.parse_args(argv)
    return parser


class ResizeIfSmallerThan:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if img.width < self.size or img.height < self.size:
            img = transforms.Resize((self.size, self.size))(img)
        return img

def main(args):
    # args = parse_args(argv)

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False



    # transforms.RandomCrop(args.patch_size)
    train_transforms = transforms.Compose(
        [ResizeIfSmallerThan(256),  # 首先检查大小并在必要时进行调整
            transforms.RandomHorizontalFlip(p=0.5),  transforms.RandomVerticalFlip(p=0.5), transforms.RandomCrop(args.patch_size),  transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [ResizeIfSmallerThan(256),  # 首先检查大小并在必要时进行调整
            transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train_DOTA_train_UC_train_val", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test_sub100_512_DOTA", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )


    if args.arch == "base":
        net = Ori_TestModel(N=args.N, M=args.M, arch=args.arch)
        net = net.to(device)
    else:
        old_net = Ori_TestModel(N=args.N, M=args.M)
        old_net = old_net.to(device)

        net = TestModel(N=args.N, M=args.M, arch=args.arch)
        net = net.to(device)



    if not os.path.exists(args.savepath):
        try:
            os.mkdir(args.savepath)
        except:
            os.makedirs(args.savepath)
    writer = SummaryWriter(args.savepath)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        if args.arch != "base":
            old_net = CustomDataParallel(old_net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=8)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1800], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, gate_weight = args.gate_weight, arch = args.arch)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)

        if args.arch == "base":
            # last_epoch = checkpoint["epoch"] + 1
            # net.load_state_dict(checkpoint["state_dict"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            state_dict = torch.load(args.checkpoint)

            net.load_state_dict(state_dict)


        elif args.arch == "dwt" or args.arch == "dwt_gate":  ## 加载前面模型参数  mlkk
            ###   首次 将第二阶段 预训练的模型 迁移过来时  ， 需要将last_epoch = 0， 随后再启动， 则需要 = checkpoint
            last_epoch =   checkpoint["epoch"] + 1
            if last_epoch != 0:
                print("!!!!!!!!!!!!!  continue training !")
                net.load_state_dict(checkpoint["state_dict"])
                aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            else:
                # 加载预训练权重到旧模型
                old_net.load_state_dict(checkpoint["state_dict"])
                net = model_fuse_load(net, old_net)

            del old_net

    stemode = False ##set the pretrained flag
    if args.checkpoint and args.pretrained:
        optimizer.param_groups[0]['lr'] = args.learning_rate
        aux_optimizer.param_groups[0]['lr'] = args.aux_learning_rate
        del lr_scheduler
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=20)
        last_epoch = 0
        stemode = True

    noisequant = True
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        if epoch > 1000 or stemode:
            noisequant = False
        print("noisequant: {}, stemode:{}".format(noisequant, stemode))
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_loss, train_bpp, train_mse = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            noisequant
        )
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Train/mse', train_mse, epoch)
        writer.add_scalar('Train/bpp', train_bpp, epoch)

        loss, bpp, mse = test_epoch(epoch, test_dataloader, net, criterion)
        writer.add_scalar('Test/loss', loss, epoch)
        writer.add_scalar('Test/mse', mse, epoch)
        writer.add_scalar('Test/bpp', bpp, epoch)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            DelfileList(args.savepath, "checkpoint_last")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                filename=os.path.join(args.savepath, "checkpoint_last_{}.pth.tar".format(epoch))
            )
            if is_best:
                DelfileList(args.savepath, "checkpoint_best")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    filename=os.path.join(args.savepath, "checkpoint_best_loss_{}.pth.tar".format(epoch))
                )
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    filename=os.path.join(args.savepath, "checkpoint_best_loss.pth.tar")
                )


def model_fuse_load(mode, pre_mode):

    model = mode  # Res2Net_NRF()  #  即将要训练的模型

    pre_mode = pre_mode   #res2net50_v1b_26w_4s(pretrained=True)

    model_dict = model.state_dict()
    # print("model_dict", model_dict)
    pretrained_dict = pre_mode.state_dict()
    # print("pretrained_dict para", pretrained_dict)
    # for k in pretrained_dict.keys():
    #   print(k)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # print("model_dict ", model_dict)
    model.load_state_dict(model_dict)

    return model

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
