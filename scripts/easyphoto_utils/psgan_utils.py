#!/usr/bin/python
# -*- encoding: utf-8 -*-
import math
import os.path as osp
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.nn import Parameter, functional
from torchvision import transforms
from torchvision.transforms import ToPILImage

pwd = osp.split(osp.realpath(__file__))[0]


# Preprocess part
def to_var(x, requires_grad=True):
    if requires_grad:
        return Variable(x).float()
    else:
        return Variable(x, requires_grad=requires_grad).float()


def copy_area(tar, src, lms):
    rect = [
        int(min(lms[:, 1])) - PreProcess.eye_margin,
        int(min(lms[:, 0])) - PreProcess.eye_margin,
        int(max(lms[:, 1])) + PreProcess.eye_margin + 1,
        int(max(lms[:, 0])) + PreProcess.eye_margin + 1,
    ]
    tar[:, :, rect[1] : rect[3], rect[0] : rect[2]] = src[:, :, rect[1] : rect[3], rect[0] : rect[2]]
    src[:, :, rect[1] : rect[3], rect[0] : rect[2]] = 0


class rectangle:
    def __init__(self, left, top, right, bottom):
        self.left_num = left
        self.top_num = top
        self.right_num = right
        self.bottom_num = bottom

    def left(self):
        return self.left_num

    def top(self):
        return self.top_num

    def right(self):
        return self.right_num

    def bottom(self):
        return self.bottom_num

    def height(self):
        return self.bottom_num - self.top_num

    def width(self):
        return self.right_num - self.left_num


def crop(image: Image, face, up_ratio, down_ratio, width_ratio):
    width, height = image.size
    face_height = face.height()
    face.width()
    delta_up = up_ratio * face_height
    delta_down = down_ratio * face_height
    delta_width = width_ratio * width

    img_left = int(max(0, face.left() - delta_width))
    img_top = int(max(0, face.top() - delta_up))
    img_right = int(min(width, face.right() + delta_width))
    img_bottom = int(min(height, face.bottom() + delta_down))
    image = image.crop((img_left, img_top, img_right, img_bottom))

    face = rectangle(face.left() - img_left, face.top() - img_top, face.right() - img_left, face.bottom() - img_top)

    center = [(img_right - img_left) / 2, (img_bottom - img_top) / 2]
    width, height = image.size
    # import ipdb; ipdb.set_trace()
    crop_left = img_left
    crop_top = img_top
    crop_right = img_right
    crop_bottom = img_bottom
    if width > height:
        left = int(center[0] - height / 2)
        right = int(center[0] + height / 2)
        if left < 0:
            left, right = 0, height
        elif right > width:
            left, right = width - height, width
        image = image.crop((left, 0, right, height))
        face = rectangle(face.left() - left, face.top(), face.right() - left, face.bottom())
        crop_left += left
        crop_right = crop_left + height
    elif width < height:
        top = int(center[1] - width / 2)
        bottom = int(center[1] + width / 2)
        if top < 0:
            top, bottom = 0, width
        elif bottom > height:
            top, bottom = height - width, height
        image = image.crop((0, top, width, bottom))
        face = rectangle(face.left(), face.top() - top, face.right(), face.bottom() - top)
        crop_top += top
        crop_bottom = crop_top + width
    crop_face = rectangle(crop_left, crop_top, crop_right, crop_bottom)
    return image, face, crop_face


class FaceParser:
    def __init__(self, device="cpu", face_skin=None):
        mapper = [0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0]
        self.device = device
        self.dic = torch.tensor(mapper, device=device)
        self.net = face_skin.model
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def parse(self, image: Image):
        assert image.shape[:2] == (512, 512)
        with torch.no_grad():
            image = self.to_tensor(image).to(self.device)
            image = torch.unsqueeze(image, 0)
            out = self.net(image)[0]
            parsing = out.squeeze(0).argmax(0)
        mask = torch.zeros_like(parsing)
        for index, num in enumerate(self.dic):
            mask[parsing == index] = num
        return mask.float()


class WingLoss(nn.Module):
    def __init__(self, wing_w=10.0, wing_epsilon=2.0):
        super(WingLoss, self).__init__()
        self.wing_w = wing_w
        self.wing_epsilon = wing_epsilon
        self.wing_c = self.wing_w * (1.0 - math.log(1.0 + self.wing_w / self.wing_epsilon))

    def forward(self, targets, predictions, euler_angle_weights=None):
        abs_error = torch.abs(targets - predictions)
        loss = torch.where(
            torch.le(abs_error, self.wing_w), self.wing_w * torch.log(1.0 + abs_error / self.wing_epsilon), abs_error - self.wing_c
        )
        loss_sum = torch.sum(loss, 1)
        if euler_angle_weights is not None:
            loss_sum *= euler_angle_weights
        return torch.mean(loss_sum)


class LinearBottleneck(nn.Module):
    def __init__(self, input_channels, out_channels, expansion, stride=1, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.expansion_channels = input_channels * expansion

        self.conv1 = nn.Conv2d(input_channels, self.expansion_channels, stride=1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.expansion_channels)

        self.depth_conv2 = nn.Conv2d(
            self.expansion_channels, self.expansion_channels, stride=stride, kernel_size=3, groups=self.expansion_channels, padding=1
        )
        self.bn2 = nn.BatchNorm2d(self.expansion_channels)

        self.conv3 = nn.Conv2d(self.expansion_channels, out_channels, stride=1, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.activation = activation(inplace=True)  # inplace=True
        self.stride = stride
        self.input_channels = input_channels
        self.out_channels = out_channels

    def forward(self, input):
        residual = input
        out = self.conv1(input)
        out = self.bn1(out)
        # out = self.activation(out)

        out = self.depth_conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.input_channels == self.out_channels:
            out += residual
        return out


class AuxiliaryNet(nn.Module):
    def __init__(self, input_channels, nums_class=3, activation=nn.ReLU, first_conv_stride=2):
        super(AuxiliaryNet, self).__init__()
        self.input_channels = input_channels
        # self.num_channels = [128, 128, 32, 128, 32]
        self.num_channels = [512, 512, 512, 512, 1024]
        self.conv1 = nn.Conv2d(self.input_channels, self.num_channels[0], kernel_size=3, stride=first_conv_stride, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels[0])

        self.conv2 = nn.Conv2d(self.num_channels[0], self.num_channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels[1])

        self.conv3 = nn.Conv2d(self.num_channels[1], self.num_channels[2], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels[2])

        self.conv4 = nn.Conv2d(self.num_channels[2], self.num_channels[3], kernel_size=7, stride=1, padding=3)
        self.bn4 = nn.BatchNorm2d(self.num_channels[3])

        self.fc1 = nn.Linear(in_features=self.num_channels[3], out_features=self.num_channels[4])
        self.fc2 = nn.Linear(in_features=self.num_channels[4], out_features=nums_class)

        self.activation = activation(inplace=True)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.activation(out)

        out = functional.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
        out = self.fc1(out)
        euler_angles_pre = self.fc2(out)

        return euler_angles_pre


class MobileNetV2(nn.Module):
    def __init__(self, input_channels=3, num_of_channels=None, nums_class=136, activation=nn.ReLU6):
        super(MobileNetV2, self).__init__()
        assert num_of_channels is not None
        self.num_of_channels = num_of_channels
        self.conv1 = nn.Conv2d(input_channels, self.num_of_channels[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_of_channels[0])

        self.depth_conv2 = nn.Conv2d(
            self.num_of_channels[0], self.num_of_channels[0], kernel_size=3, stride=1, padding=1, groups=self.num_of_channels[0]
        )
        self.bn2 = nn.BatchNorm2d(self.num_of_channels[0])

        self.stage0 = self.make_stage(
            self.num_of_channels[0], self.num_of_channels[0], stride=2, stage=0, times=5, expansion=2, activation=activation
        )

        self.stage1 = self.make_stage(
            self.num_of_channels[0], self.num_of_channels[1], stride=2, stage=1, times=7, expansion=4, activation=activation
        )

        self.linear_bottleneck_end = nn.Sequential(
            LinearBottleneck(self.num_of_channels[1], self.num_of_channels[2], expansion=2, stride=1, activation=activation)
        )

        self.conv3 = nn.Conv2d(self.num_of_channels[2], self.num_of_channels[3], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_of_channels[3])

        self.conv4 = nn.Conv2d(self.num_of_channels[3], self.num_of_channels[4], kernel_size=7, stride=1)
        self.bn4 = nn.BatchNorm2d(self.num_of_channels[4])

        self.activation = activation(inplace=True)

        self.in_features = 14 * 14 * self.num_of_channels[2] + 7 * 7 * self.num_of_channels[3] + 1 * 1 * self.num_of_channels[4]
        self.fc = nn.Linear(in_features=self.in_features, out_features=nums_class)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_stage(self, input_channels, out_channels, stride, stage, times, expansion, activation=nn.ReLU6):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        module = LinearBottleneck(input_channels, out_channels, expansion=2, stride=stride, activation=activation)
        modules[stage_name + "_0"] = module

        for i in range(times - 1):
            module = LinearBottleneck(out_channels, out_channels, expansion=expansion, stride=1, activation=activation)
            module_name = stage_name + "_{}".format(i + 1)
            modules[module_name] = module

        return nn.Sequential(modules)

    def forward(self, input):
        with torch.no_grad():
            out = self.conv1(input)
            out = self.bn1(out)
            out = self.activation(out)

            out = self.depth_conv2(out)
            out = self.bn2(out)
            out = self.activation(out)

            out = self.stage0(out)
            out1 = self.stage1(out)

            out1 = self.linear_bottleneck_end(out1)

            out2 = self.conv3(out1)
            out2 = self.bn3(out2)
            out2 = self.activation(out2)

            out3 = self.conv4(out2)
            out3 = self.bn4(out3)
            out3 = self.activation(out3)

            out1 = out1.contiguous().view(out1.size(0), -1)
            out2 = out2.contiguous().view(out2.size(0), -1)
            out3 = out3.contiguous().view(out3.size(0), -1)

            multi_scale = torch.cat([out1, out2, out3], 1)
            assert multi_scale.size(1) == self.in_features
            pre_landmarks = self.fc(multi_scale)
        return pre_landmarks, out


class PreProcess:
    eye_margin = 16
    diff_size = (64, 64)

    def __init__(self, device="cpu", need_parser=True, retinaface_detection=None, face_skin=None, landmark_path=None):
        self.device = device
        self.img_size = 256

        xs, ys = np.meshgrid(np.linspace(0, self.img_size - 1, self.img_size), np.linspace(0, self.img_size - 1, self.img_size))
        xs = xs[None].repeat(68, axis=0)
        ys = ys[None].repeat(68, axis=0)
        fix = np.concatenate([ys, xs], axis=0)
        self.fix = torch.Tensor(fix).to(self.device)
        self.retinaface_detection = retinaface_detection
        if need_parser:
            self.face_parse = FaceParser(device=device, face_skin=face_skin)

        self.landmark = MobileNetV2(num_of_channels=[64, 128, 16, 32, 128], nums_class=136)
        self.landmark.load_state_dict(torch.load(landmark_path))
        self.landmark.eval().to(self.device)
        self.up_ratio = 0.6 / 0.85
        self.down_ratio = 0.2 / 0.85
        self.width_ratio = 0.2 / 0.85
        self.lip_class = [7, 9]
        self.face_class = [1, 6]

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def relative2absolute(self, lms):
        return lms * self.img_size

    def process(self, mask, lms, device="cpu"):
        diff = to_var(
            (self.fix.double() - torch.tensor(lms.transpose((1, 0)).reshape(-1, 1, 1)).to(self.device)).unsqueeze(0), requires_grad=False
        ).to(self.device)

        lms_eye_left = lms[42:48]
        lms_eye_right = lms[36:42]
        lms = lms.transpose((1, 0)).reshape(-1, 1, 1)  # transpose to (y-x)
        # lms = np.tile(lms, (1, 256, 256))  # (136, h, w)
        diff = to_var((self.fix.double() - torch.tensor(lms).to(self.device)).unsqueeze(0), requires_grad=False).to(self.device)

        mask_lip = (mask == self.lip_class[0]).float() + (mask == self.lip_class[1]).float()
        mask_face = (mask == self.face_class[0]).float() + (mask == self.face_class[1]).float()

        mask_eyes = torch.zeros_like(mask, device=device)
        copy_area(mask_eyes, mask_face, lms_eye_left)
        copy_area(mask_eyes, mask_face, lms_eye_right)
        mask_eyes = to_var(mask_eyes, requires_grad=False).to(device)

        mask_list = [mask_lip, mask_face, mask_eyes]
        mask_aug = torch.cat(mask_list, 0)  # (3, 1, h, w)
        mask_re = F.interpolate(mask_aug, size=self.diff_size).repeat(1, diff.shape[1], 1, 1)  # (3, 136, 64, 64)
        diff_re = F.interpolate(diff, size=self.diff_size).repeat(3, 1, 1, 1)  # (3, 136, 64, 64)
        diff_re = diff_re * mask_re  # (3, 136, 32, 32)
        norm = torch.norm(diff_re, dim=1, keepdim=True).repeat(1, diff_re.shape[1], 1, 1)
        norm = torch.where(norm == 0, torch.tensor(1e10, device=device), norm)
        diff_re /= norm

        return mask_aug, diff_re

    def __call__(self, image: Image):
        retinaface_result = self.retinaface_detection(image)
        face = []
        for box in retinaface_result["boxes"]:
            face.append(rectangle(*np.int32(box)))

        if len(face) == 0:
            return None, None, None

        face_on_image = face[0]
        image, face, crop_face = crop(image, face_on_image, self.up_ratio, self.down_ratio, self.width_ratio)
        np_image = np.array(image)
        mask = self.face_parse.parse(cv2.resize(np_image, (512, 512)))

        # obtain face parsing result
        mask = F.interpolate(mask.view(1, 1, 512, 512), (self.img_size, self.img_size), mode="nearest")
        mask = mask.type(torch.uint8)
        mask = to_var(mask, requires_grad=False).to(self.device)

        input = image.crop([face.left(), face.top(), face.right(), face.bottom()])
        input = input.resize([112, 112])
        input = np.expand_dims(np.array(input, np.float32) / 255.0, 0)
        input = torch.Tensor(input.transpose((0, 3, 1, 2))).to(self.device)

        pre_landmarks, _ = self.landmark(input)
        lms = pre_landmarks[0].cpu().detach().numpy()
        lms = lms.reshape(-1, 2) * [face.width(), face.height()] + np.int32([face.left(), face.top()])
        lms = lms / [np.shape(image)[0], np.shape(image)[1]] * self.img_size
        lms = lms[:, ::-1]

        mask, diff = self.process(mask, lms, device=self.device)
        image = image.resize((self.img_size, self.img_size), Image.ANTIALIAS)
        image = self.transform(image)
        real = to_var(image.unsqueeze(0))
        return [real, mask, diff], face_on_image, crop_face


# Solver part (GAN part)
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(object):
    def __init__(self):
        self.name = "weight"
        self.power_iterations = 1

    def compute_weight(self, module):
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        w = getattr(module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        return w / sigma.expand_as(w)

    @staticmethod
    def apply(module):
        name = "weight"
        fn = SpectralNorm()

        try:
            u = getattr(module, name + "_u")
            v = getattr(module, name + "_v")
            w = getattr(module, name + "_bar")
        except AttributeError:
            w = getattr(module, name)
            height = w.data.shape[0]
            width = w.view(height, -1).data.shape[1]
            u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
            v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
            w_bar = Parameter(w.data)

            # del module._parameters[name]

            module.register_parameter(name + "_u", u)
            module.register_parameter(name + "_v", v)
            module.register_parameter(name + "_bar", w_bar)

        # remove w from parameter list
        del module._parameters[name]

        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + "_u"]
        del module._parameters[self.name + "_v"]
        del module._parameters[self.name + "_bar"]
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def spectral_norm(module):
    SpectralNorm.apply(module)
    return module


def remove_spectral_norm(module):
    name = "weight"
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class ResidualBlock(nn.Module):
    """Residual Block."""

    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == "p" or (net_mode is None):
            use_affine = True
        elif net_mode == "t":
            use_affine = False
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine),
        )

    def forward(self, x):
        return x + self.main(x)


class GetMatrix(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GetMatrix, self).__init__()
        self.get_gamma = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.get_beta = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        gamma = self.get_gamma(x)
        beta = self.get_beta(x)
        return x, gamma, beta


class NONLocalBlock2D(nn.Module):
    def __init__(self):
        super(NONLocalBlock2D, self).__init__()
        self.g = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, source, weight):
        """(b, c, h, w)
        src_diff: (3, 136, 32, 32)
        """
        batch_size = source.size(0)

        g_source = source.view(batch_size, 1, -1)  # (N, C, H*W)
        g_source = g_source.permute(0, 2, 1)  # (N, H*W, C)

        y = torch.bmm(weight.to_dense(), g_source)
        y = y.permute(0, 2, 1).contiguous()  # (N, C, H*W)
        y = y.view(batch_size, 1, *source.size()[2:])
        return y


class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    def __init__(self):
        super(Generator, self).__init__()

        # -------------------------- PNet(MDNet) for obtaining makeup matrices --------------------------

        layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False), nn.InstanceNorm2d(64, affine=True), nn.ReLU(inplace=True)
        )
        self.pnet_in = layers

        # Down-Sampling
        curr_dim = 64
        for i in range(2):
            layers = nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(inplace=True),
            )

            setattr(self, f"pnet_down_{i+1}", layers)
            curr_dim = curr_dim * 2

        # Bottleneck. All bottlenecks share the same attention module
        self.atten_bottleneck_g = NONLocalBlock2D()
        self.atten_bottleneck_b = NONLocalBlock2D()
        self.simple_spade = GetMatrix(curr_dim, 1)  # get the makeup matrix

        for i in range(3):
            setattr(self, f"pnet_bottleneck_{i+1}", ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode="p"))

        # --------------------------- TNet(MANet) for applying makeup transfer ----------------------------

        self.tnet_in_conv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.tnet_in_spade = nn.InstanceNorm2d(64, affine=False)
        self.tnet_in_relu = nn.ReLU(inplace=True)

        # Down-Sampling
        curr_dim = 64
        for i in range(2):
            setattr(self, f"tnet_down_conv_{i+1}", nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            setattr(self, f"tnet_down_spade_{i+1}", nn.InstanceNorm2d(curr_dim * 2, affine=False))
            setattr(self, f"tnet_down_relu_{i+1}", nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(6):
            setattr(self, f"tnet_bottleneck_{i+1}", ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode="t"))

        # Up-Sampling
        for i in range(2):
            setattr(
                self, f"tnet_up_conv_{i+1}", nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False)
            )
            setattr(self, f"tnet_up_spade_{i+1}", nn.InstanceNorm2d(curr_dim // 2, affine=False))
            setattr(self, f"tnet_up_relu_{i+1}", nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers = nn.Sequential(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False), nn.Tanh())
        self.tnet_out = layers

    @staticmethod
    def atten_feature(mask_s, weight, gamma_s, beta_s, atten_module_g, atten_module_b):
        """
        feature size: (1, c, h, w)
        mask_c(s): (3, 1, h, w)
        diff_c: (1, 138, 256, 256)
        return: (1, c, h, w)
        """
        channel_num = gamma_s.shape[1]

        mask_s_re = F.interpolate(mask_s, size=gamma_s.shape[2:]).repeat(1, channel_num, 1, 1)
        gamma_s_re = gamma_s.repeat(3, 1, 1, 1)
        gamma_s = gamma_s_re * mask_s_re  # (3, c, h, w)
        beta_s_re = beta_s.repeat(3, 1, 1, 1)
        beta_s = beta_s_re * mask_s_re

        gamma = atten_module_g(gamma_s, weight)  # (3, c, h, w)
        beta = atten_module_b(beta_s, weight)

        gamma = (gamma[0] + gamma[1] + gamma[2]).unsqueeze(0)  # (c, h, w) combine the three parts
        beta = (beta[0] + beta[1] + beta[2]).unsqueeze(0)
        return gamma, beta

    def get_weight(self, mask_c, mask_s, fea_c, fea_s, diff_c, diff_s):
        """s --> source; c --> target
        feature size: (1, 256, 64, 64)
        diff: (3, 136, 32, 32)
        """
        HW = 64 * 64
        batch_size = 3
        assert fea_s is not None  # fea_s when i==3
        # get 3 part fea using mask
        channel_num = fea_s.shape[1]

        mask_c_re = F.interpolate(mask_c, size=64).repeat(1, channel_num, 1, 1)  # (3, c, h, w)
        fea_c = fea_c.repeat(3, 1, 1, 1)  # (3, c, h, w)
        fea_c = fea_c * mask_c_re  # (3, c, h, w) 3 stands for 3 parts

        mask_s_re = F.interpolate(mask_s, size=64).repeat(1, channel_num, 1, 1)
        fea_s = fea_s.repeat(3, 1, 1, 1)
        fea_s = fea_s * mask_s_re

        theta_input = torch.cat((fea_c * 0.01, diff_c), dim=1)
        phi_input = torch.cat((fea_s * 0.01, diff_s), dim=1)

        theta_target = theta_input.view(batch_size, -1, HW)  # (N, C+136, H*W)
        theta_target = theta_target.permute(0, 2, 1)  # (N, H*W, C+136)

        phi_source = phi_input.view(batch_size, -1, HW)  # (N, C+136, H*W)

        weight = torch.bmm(theta_target, phi_source)  # (3, HW, HW)
        with torch.no_grad():
            v = weight.detach().nonzero().long().permute(1, 0)
            # This clone is required to correctly release cuda memory.
            weight_ind = v.clone()
            del v
            torch.cuda.empty_cache()

        weight *= 200  # hyper parameters for visual feature
        weight = F.softmax(weight, dim=-1)
        weight = weight[weight_ind[0], weight_ind[1], weight_ind[2]]
        ret = torch.sparse.FloatTensor(weight_ind, weight, torch.Size([3, HW, HW]))
        return ret

    def forward(self, c, s, mask_c, mask_s, diff_c, diff_s, gamma=None, beta=None, ret=False):
        c, s, mask_c, mask_s, diff_c, diff_s = [x.squeeze(0) if x.ndim == 5 else x for x in [c, s, mask_c, mask_s, diff_c, diff_s]]
        """attention version
        c: content, stands for source image. shape: (b, c, h, w)
        s: style, stands for reference image. shape: (b, c, h, w)
        mask_list_c: lip, skin, eye. (b, 1, h, w)
        """

        # forward c in tnet(MANet)
        c_tnet = self.tnet_in_conv(c)
        s = self.pnet_in(s)
        c_tnet = self.tnet_in_spade(c_tnet)
        c_tnet = self.tnet_in_relu(c_tnet)

        # down-sampling
        for i in range(2):
            if gamma is None:
                cur_pnet_down = getattr(self, f"pnet_down_{i+1}")
                s = cur_pnet_down(s)

            cur_tnet_down_conv = getattr(self, f"tnet_down_conv_{i+1}")
            cur_tnet_down_spade = getattr(self, f"tnet_down_spade_{i+1}")
            cur_tnet_down_relu = getattr(self, f"tnet_down_relu_{i+1}")
            c_tnet = cur_tnet_down_conv(c_tnet)
            c_tnet = cur_tnet_down_spade(c_tnet)
            c_tnet = cur_tnet_down_relu(c_tnet)

        # bottleneck
        for i in range(6):
            if gamma is None and i <= 2:
                cur_pnet_bottleneck = getattr(self, f"pnet_bottleneck_{i+1}")
            cur_tnet_bottleneck = getattr(self, f"tnet_bottleneck_{i+1}")

            # get s_pnet from p and transform
            if i == 3:
                if gamma is None:  # not in test_mix
                    s, gamma, beta = self.simple_spade(s)
                    weight = self.get_weight(mask_c, mask_s, c_tnet, s, diff_c, diff_s)
                    gamma, beta = self.atten_feature(mask_s, weight, gamma, beta, self.atten_bottleneck_g, self.atten_bottleneck_b)
                    if ret:
                        return [gamma, beta]
                # else:                       # in test mode
                # gamma, beta = param_A[0]*w + param_B[0]*(1-w), param_A[1]*w + param_B[1]*(1-w)

                c_tnet = c_tnet * (1 + gamma) + beta  # apply makeup transfer using makeup matrices

            if gamma is None and i <= 2:
                s = cur_pnet_bottleneck(s)
            c_tnet = cur_tnet_bottleneck(c_tnet)

        # up-sampling
        for i in range(2):
            cur_tnet_up_conv = getattr(self, f"tnet_up_conv_{i+1}")
            cur_tnet_up_spade = getattr(self, f"tnet_up_spade_{i+1}")
            cur_tnet_up_relu = getattr(self, f"tnet_up_relu_{i+1}")
            c_tnet = cur_tnet_up_conv(c_tnet)
            c_tnet = cur_tnet_up_spade(c_tnet)
            c_tnet = cur_tnet_up_relu(c_tnet)

        c_tnet = self.tnet_out(c_tnet)
        return c_tnet


# Gan Solver
class Solver:
    def __init__(self, device="cpu", inference=None):
        self.G = Generator()
        self.G.load_state_dict(torch.load(inference, map_location=torch.device(device)))
        self.G = self.G.to(device).eval()
        return

    def generate(
        self, org_A, ref_B, lms_A=None, lms_B=None, mask_A=None, mask_B=None, diff_A=None, diff_B=None, gamma=None, beta=None, ret=False
    ):
        """org_A is content, ref_B is style"""
        res = self.G(org_A, ref_B, mask_A, mask_B, diff_A, diff_B, gamma, beta, ret)
        return res

    def test(self, real_A, mask_A, diff_A, real_B, mask_B, diff_B):
        cur_prama = None
        with torch.no_grad():
            cur_prama = self.generate(real_A, real_B, None, None, mask_A, mask_B, diff_A, diff_B, ret=True)
            fake_A = self.generate(real_A, real_B, None, None, mask_A, mask_B, diff_A, diff_B, gamma=cur_prama[0], beta=cur_prama[1])
        fake_A = fake_A.squeeze(0)

        # normalize
        min_, max_ = fake_A.min(), fake_A.max()
        fake_A.add_(-min_).div_(max_ - min_ + 1e-5)

        return ToPILImage()(fake_A.cpu())


# PostProcess part
class PostProcess:
    def __init__(self):
        self.denoise = False
        self.img_size = 256

    def __call__(self, source: Image, result: Image):
        source = np.array(source)
        result = np.array(result)

        height, width = source.shape[:2]
        small_source = cv2.resize(source, (self.img_size, self.img_size))
        laplacian_diff = source.astype(np.float64) - cv2.resize(small_source, (width, height)).astype(np.float64)
        result = (cv2.resize(result, (width, height)) + laplacian_diff).round().clip(0, 255).astype(np.uint8)
        if self.denoise:
            result = cv2.fastNlMeansDenoisingColored(result)
        result = Image.fromarray(result).convert("RGB")
        return result


class PSGAN_Inference:
    """
    An inference wrapper for makeup transfer.
    It takes two image `source` and `reference` in,
    and transfers the makeup of reference to source.
    """

    def __init__(self, device="cpu", model_path="assets/models/G.pth", retinaface_detection=None, face_skin=None, landmark_path=None):
        """
        Args:
            device (str): Device type and index, such as "cpu" or "cuda:2".
            device_id (int): Specifying which device index
                will be used for inference.
        """
        self.device = device
        self.solver = Solver(device, inference=model_path)
        self.preprocess = PreProcess(device, retinaface_detection=retinaface_detection, face_skin=face_skin, landmark_path=landmark_path)
        self.postprocess = PostProcess()

    def transfer(self, source: Image, reference: Image):
        """
        Args:
            source (Image): The image where makeup will be transferred to.
            reference (Image): Image containing targeted makeup.
        Return:
            Image: Transferred image.
        """
        source_input, face, crop_face = self.preprocess(source)
        reference_input, _, _ = self.preprocess(reference)

        if not (source_input and reference_input):
            return source

        for i in range(len(source_input)):
            source_input[i] = source_input[i].to(self.device)

        for i in range(len(reference_input)):
            reference_input[i] = reference_input[i].to(self.device)

        # TODO: Abridge the parameter list.
        result = self.solver.test(*source_input, *reference_input)

        source_crop = source.crop((crop_face.left(), crop_face.top(), crop_face.right(), crop_face.bottom()))
        result = self.postprocess(source_crop, result)
        return result
