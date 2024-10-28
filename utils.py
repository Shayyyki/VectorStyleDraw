#@title Load CLIP {vertical-output: true}

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR100
import os
import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import requests
import numpy as np
import matplotlib.pylab as pl
import glob
import copy
from tqdm import tqdm
from torchvision import utils
import torch
from skimage.transform import resize
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# print("Torch version:", torch.__version__)
from PIL import Image
import pydiffvg
import skimage
import skimage.io
import random
# import ttools.modules
import argparse
import math
import torchvision
import torchvision.transforms as transforms
import requests
from io import BytesIO
from IPython.display import display

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import PIL
from time import time
import config
from IPython.display import Image, HTML, clear_output
from tqdm import tqdm_notebook, tnrange
# from U2Net_.model import U2NET
# os.environ['FFMPEG_BINARY'] = 'ffmpeg'
# import moviepy.editor as mvp
# from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
# device = torch.device("cuda:0")
device = torch.device("cuda:0")
args = config.parse_arguments()
args.device = device
def imread(url, max_size=None, mode=None):
  if url.startswith(('http:', 'https:')):
    r = requests.get(url)
    f = io.BytesIO(r.content)
  else:
    f = url
  img = PIL.Image.open(f)
  if max_size is not None:
    img = img.resize((max_size, max_size))
  if mode is not None:
    img = img.convert(mode)
  img = np.float32(img)/255.0
  return img

def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

def im2url(a, fmt='jpeg'):
  encoded = imencode(a, fmt)
  base64_byte_string = base64.b64encode(encoded).decode('ascii')
  return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

def imshow(a, fmt='jpeg'):
  display(Image(data=imencode(a, fmt)))

def tile2d(a, w=None):
  a = np.asarray(a)
  if w is None:
    w = int(np.ceil(np.sqrt(len(a))))
  th, tw = a.shape[1:3]
  pad = (w-len(a))%w
  a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
  h = len(a)//w
  a = a.reshape([h, w]+list(a.shape[1:]))
  a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
  return a

def show_img(img):
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = np.uint8(img * 254)
    # img = np.repeat(img, 4, axis=0)
    # img = np.repeat(img, 4, axis=1)
    pimg = PIL.Image.fromarray(img, mode="RGB")
    imshow(pimg)

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img

class VideoWriter:
  def __init__(self, filename='_autoplay.mp4', fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()
    if self.params['filename'] == '_autoplay.mp4':
      self.show()

  def show(self, **kw):
      self.close()
      fn = self.params['filename']
      display(mvp.ipython_display(fn, **kw))

# @title Style Loss and Drawing Functions {vertical-output: true}
def pil_resize_long_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_long = (trg_size / pil.height) if short_w else (trg_size / pil.width)
    resized = pil.resize((int(pil.width * ar_resized_long), int(pil.height * ar_resized_long)), PIL.Image.BICUBIC)
    return resized


class Vgg16_Extractor(nn.Module):
    def __init__(self, space):
        super().__init__()
        self.vgg_layers = models.vgg16(pretrained=True).features

        for param in self.parameters():
            param.requires_grad = False
        self.capture_layers = [1, 3, 6, 8, 11, 13, 15, 22, 29]
        self.space = space

    def forward_base(self, x):
        feat = [x]
        for i in range(max(self.capture_layers) + 1):
            x = self.vgg_layers[i](x)
            if i in self.capture_layers: feat.append(x)
        return feat

    def forward(self, x):
        if self.space != 'vgg':
            x = (x + 1.) / 2.
            x = x - (torch.Tensor([0.485, 0.456, 0.406]).to(x.device).view(1, -1, 1, 1))
            x = x / (torch.Tensor([0.229, 0.224, 0.225]).to(x.device).view(1, -1, 1, 1))
        feat = self.forward_base(x)
        return feat

    def forward_samples_hypercolumn(self, X, samps=100):
        feat = self.forward(X)

        xx, xy = np.meshgrid(np.arange(X.shape[2]), np.arange(X.shape[3]))
        xx = np.expand_dims(xx.flatten(), 1)
        xy = np.expand_dims(xy.flatten(), 1)
        xc = np.concatenate([xx, xy], 1)

        samples = min(samps, xc.shape[0])

        np.random.shuffle(xc)
        xx = xc[:samples, 0]
        yy = xc[:samples, 1]

        feat_samples = []
        for i in range(len(feat)):

            layer_feat = feat[i]

            # hack to detect lower resolution
            if i > 0 and feat[i].size(2) < feat[i - 1].size(2):
                xx = xx / 2.0
                yy = yy / 2.0

            xx = np.clip(xx, 0, layer_feat.shape[2] - 1).astype(np.int32)
            yy = np.clip(yy, 0, layer_feat.shape[3] - 1).astype(np.int32)

            features = layer_feat[:, :, xx[range(samples)], yy[range(samples)]]
            feat_samples.append(features.clone().detach())

        feat = torch.cat(feat_samples, 1)
        return feat


# Tensor and PIL utils

def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')


def pil_loader_internet(url):
    response = requests.get(url)
    img = PIL.Image.open(BytesIO(response.content))
    return img.convert('RGB')


def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)


def pil_resize_short_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_short = (trg_size / pil.width) if short_w else (trg_size / pil.height)
    resized = pil.resize((int(pil.width * ar_resized_short), int(pil.height * ar_resized_short)), PIL.Image.BICUBIC)
    return resized


def pil_resize_long_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_long = (trg_size / pil.height) if short_w else (trg_size / pil.width)
    resized = pil.resize((int(pil.width * ar_resized_long), int(pil.height * ar_resized_long)), PIL.Image.BICUBIC)
    return resized


def np_to_pil(npy):
    return PIL.Image.fromarray(npy.astype(np.uint8))


def pil_to_np(pil):
    return np.array(pil)


def tensor_to_np(tensor, cut_dim_to_3=True):
    if len(tensor.shape) == 4:
        if cut_dim_to_3:
            tensor = tensor[0]
        else:
            return tensor.data.cpu().numpy().transpose((0, 2, 3, 1))
    return tensor.data.cpu().numpy().transpose((1, 2, 0))


def np_to_tensor(npy, space):
    if space == 'vgg':
        return np_to_tensor_correct(npy)
    return (torch.Tensor(npy.astype(np.float) / 127.5) - 1.0).permute((2, 0, 1)).unsqueeze(0)


def np_to_tensor_correct(npy):
    pil = np_to_pil(npy)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    return transform(pil).unsqueeze(0)


# Laplacian Pyramid

def laplacian(x):
    # x - upsample(downsample(x))
    return x - tensor_resample(tensor_resample(x, [x.shape[2] // 2, x.shape[3] // 2]), [x.shape[2], x.shape[3]])


def make_laplace_pyramid(x, levels):
    pyramid = []
    current = x
    for i in range(levels):
        pyramid.append(laplacian(current))
        current = tensor_resample(current, (max(current.shape[2] // 2, 1), max(current.shape[3] // 2, 1)))
    pyramid.append(current)
    return pyramid


def fold_laplace_pyramid(pyramid):
    current = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):  # iterate from len-2 to 0
        up_h, up_w = pyramid[i].shape[2], pyramid[i].shape[3]
        current = pyramid[i] + tensor_resample(current, (up_h, up_w))
    return current


def sample_indices(feat_content, feat_style):
    indices = None
    const = 128 ** 2  # 32k or so
    feat_dims = feat_style.shape[1]
    big_size = feat_content.shape[2] * feat_content.shape[3]  # num feaxels

    stride_x = int(max(math.floor(math.sqrt(big_size // const)), 1))
    offset_x = np.random.randint(stride_x)
    stride_y = int(max(math.ceil(math.sqrt(big_size // const)), 1))
    offset_y = np.random.randint(stride_y)
    xx, xy = np.meshgrid(np.arange(feat_content.shape[2])[offset_x::stride_x],
                         np.arange(feat_content.shape[3])[offset_y::stride_y])

    xx = xx.flatten()
    xy = xy.flatten()
    return xx, xy


def spatial_feature_extract(feat_result, feat_content, xx, xy):
    l2, l3 = [], []
    device = feat_result[0].device

    # for each extracted layer
    for i in range(len(feat_result)):
        fr = feat_result[i]
        fc = feat_content[i]

        # hack to detect reduced scale
        if i > 0 and feat_result[i - 1].size(2) > feat_result[i].size(2):
            xx = xx / 2.0
            xy = xy / 2.0

        # go back to ints and get residual
        xxm = np.floor(xx).astype(np.float32)
        xxr = xx - xxm

        xym = np.floor(xy).astype(np.float32)
        xyr = xy - xym

        # do bilinear resample
        w00 = torch.from_numpy((1. - xxr) * (1. - xyr)).float().view(1, 1, -1, 1).to(device)
        w01 = torch.from_numpy((1. - xxr) * xyr).float().view(1, 1, -1, 1).to(device)
        w10 = torch.from_numpy(xxr * (1. - xyr)).float().view(1, 1, -1, 1).to(device)
        w11 = torch.from_numpy(xxr * xyr).float().view(1, 1, -1, 1).to(device)

        xxm = np.clip(xxm.astype(np.int32), 0, fr.size(2) - 1)
        xym = np.clip(xym.astype(np.int32), 0, fr.size(3) - 1)

        s00 = xxm * fr.size(3) + xym
        s01 = xxm * fr.size(3) + np.clip(xym + 1, 0, fr.size(3) - 1)
        s10 = np.clip(xxm + 1, 0, fr.size(2) - 1) * fr.size(3) + (xym)
        s11 = np.clip(xxm + 1, 0, fr.size(2) - 1) * fr.size(3) + np.clip(xym + 1, 0, fr.size(3) - 1)

        fr = fr.view(1, fr.size(1), fr.size(2) * fr.size(3), 1)
        fr = fr[:, :, s00, :].mul_(w00).add_(fr[:, :, s01, :].mul_(w01)).add_(fr[:, :, s10, :].mul_(w10)).add_(
            fr[:, :, s11, :].mul_(w11))

        fc = fc.view(1, fc.size(1), fc.size(2) * fc.size(3), 1)
        fc = fc[:, :, s00, :].mul_(w00).add_(fc[:, :, s01, :].mul_(w01)).add_(fc[:, :, s10, :].mul_(w10)).add_(
            fc[:, :, s11, :].mul_(w11))

        l2.append(fr)
        l3.append(fc)

    x_st = torch.cat([li.contiguous() for li in l2], 1)
    c_st = torch.cat([li.contiguous() for li in l3], 1)

    xx = torch.from_numpy(xx).view(1, 1, x_st.size(2), 1).float().to(device)
    yy = torch.from_numpy(xy).view(1, 1, x_st.size(2), 1).float().to(device)

    x_st = torch.cat([x_st, xx, yy], 1)
    c_st = torch.cat([c_st, xx, yy], 1)
    return x_st, c_st


def pairwise_distances_cos(x, y):
    x_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y ** 2).sum(1).view(1, -1))
    dist = 1. - torch.mm(x, y_t) / x_norm / y_norm
    return dist


def pairwise_distances_sq_l2(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 1e-5, 1e5) / x.size(1)


def distmat(x, y, cos_d=True):
    if cos_d:
        M = pairwise_distances_cos(x, y)
    else:
        M = torch.sqrt(pairwise_distances_sq_l2(x, y))
    return M


def content_loss(feat_result, feat_content):
    d = feat_result.size(1)

    X = feat_result.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
    Y = feat_content.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    Y = Y[:, :-2]
    X = X[:, :-2]
    # X = X.t()
    # Y = Y.t()

    Mx = distmat(X, X)
    Mx = Mx  # /Mx.sum(0, keepdim=True)

    My = distmat(Y, Y)
    My = My  # /My.sum(0, keepdim=True)

    d = torch.abs(Mx - My).mean()  # * X.shape[0]
    return d


def rgb_to_yuv(rgb):
    C = torch.Tensor(
        [[0.577350, 0.577350, 0.577350], [-0.577350, 0.788675, -0.211325], [-0.577350, -0.211325, 0.788675]]).to(
        rgb.device)
    yuv = torch.mm(C, rgb)
    return yuv


def style_loss(X, Y, cos_d=True):
    d = X.shape[1]

    if d == 3:
        X = rgb_to_yuv(X.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
        Y = rgb_to_yuv(Y.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
    else:
        X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
        Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    # Relaxed EMD
    CX_M = distmat(X, Y, cos_d=True)

    if d == 3: CX_M = CX_M + distmat(X, Y, cos_d=False)

    m1, m1_inds = CX_M.min(1)
    m2, m2_inds = CX_M.min(0)

    remd = torch.max(m1.mean(), m2.mean())

    return remd


def moment_loss(X, Y, moments=[1, 2]):
    loss = 0.
    X = X.squeeze().t()
    Y = Y.squeeze().t()

    mu_x = torch.mean(X, 0, keepdim=True)
    mu_y = torch.mean(Y, 0, keepdim=True)
    mu_d = torch.abs(mu_x - mu_y).mean()

    if 1 in moments:
        # print(mu_x.shape)
        loss = loss + mu_d

    if 2 in moments:
        X_c = X - mu_x
        Y_c = Y - mu_y
        X_cov = torch.mm(X_c.t(), X_c) / (X.shape[0] - 1)
        Y_cov = torch.mm(Y_c.t(), Y_c) / (Y.shape[0] - 1)

        # print(X_cov.shape)
        # exit(1)

        D_cov = torch.abs(X_cov - Y_cov).mean()
        loss = loss + D_cov

    return loss


def calculate_loss(feat_result, feat_content, feat_style, indices, content_weight, moment_weight=1.0):
    # spatial feature extract
    num_locations = 1024
    spatial_result, spatial_content = spatial_feature_extract(feat_result, feat_content, indices[0][:num_locations],
                                                              indices[1][:num_locations])

    # loss_content = content_loss(spatial_result, spatial_content)
    loss_content = 0
    d = feat_style.shape[1]
    spatial_style = feat_style.view(1, d, -1, 1)
    feat_max = d  # 3+2*64+128*2+256*3+512*2 # (sum of all extracted channels)

    loss_remd = style_loss(spatial_result[:, :feat_max, :, :], spatial_style[:, :feat_max, :, :])

    loss_moment = moment_loss(spatial_result[:, :-2, :, :], spatial_style, moments=[1, 2])  # -2 is so that it can fit?
    # palette matching
    content_weight_frac = 1. / max(content_weight, 1.)
    loss_moment += content_weight_frac * style_loss(spatial_result[:, :3, :, :], spatial_style[:, :3, :, :])

    loss_style = loss_remd + moment_weight * loss_moment
    # print(f'Style: {loss_style.item():.3f}, Content: {loss_content.item():.3f}')

    style_weight = 1.0 + moment_weight
    loss_total = loss_style / (content_weight + style_weight)
    # loss_total = loss_content * (content_weight / 2) + loss_style * (style_weight / 2)

    # print(f'Style_loss: {loss_style:.3f}, Content_loss: {loss_content:.3f}')
    # print(f'Style_weight: {style_weight:.3f}, Content_weight: {content_weight:.3f}')
    # print(f'total_loss: {loss_total:.3f}')
    return loss_total

def get_image_augmentation(use_normalized_clip):
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
    ])

    if use_normalized_clip:
        augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    return augment_trans


def initialize_curves(num_paths, canvas_width, canvas_height):
    shapes = []
    shape_groups = []
    for i in range(num_paths):
        num_segments = random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)
        for j in range(num_segments):
            radius = 0.1
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points=num_control_points, points=points, stroke_width=torch.tensor(1.0),
                             is_closed=False)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                         stroke_color=torch.tensor(
                                             [random.random(), random.random(), random.random(), random.random()]))
        shape_groups.append(path_group)
    return shapes, shape_groups
def save_svg(output_dir, name, canvas_width, canvas_height, shapes,shape_groups):
    pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), canvas_width, canvas_height, shapes, shape_groups)
def render_drawing(image_name, num_paths, shapes, shape_groups, \
                   canvas_width, canvas_height, n_iter, save=False):
    scene_args = pydiffvg.RenderFunction.serialize_scene( \
        canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, n_iter, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()) * (
                1 - img[:, :, 3:4])
    if save:
        pydiffvg.imwrite(img.cpu(), './content/{}_num{}/iter_{}.png'.format(image_name, num_paths, int(n_iter)), gamma=1.0)
        save_svg(f"./svg_logs", f"{image_name}_num{num_paths}_{int(n_iter)}", canvas_width, canvas_height, shapes, shape_groups)
    img = img[:, :, :3]
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
    return img


def render_scaled(image_name, num_paths, shapes, shape_groups, original_height, original_width,
                  scale_factor=4, t=1):
    '''
        Scale the size of the rendered image
    '''
    with torch.no_grad():
        shapes_resized = copy.deepcopy(shapes)
        for i in range(len(shapes)):
            shapes_resized[i].stroke_width = shapes[i].stroke_width * scale_factor
            for j in range(len(shapes[i].points)):
                shapes_resized[i].points[j] = shapes[i].points[j] * scale_factor
        img = render_drawing(image_name, num_paths, shapes_resized, shape_groups,
                             int(original_width * scale_factor), int(original_height * scale_factor), t)
        return img

def set_inds_clip(self):
    attn_map = (self.attention_map - self.attention_map.min()) / (
            self.attention_map.max() - self.attention_map.min())
    if self.xdog_intersec:
        xdog = XDoG_()
        im_xdog = xdog(self.image_input_attn_clip[0].permute(1, 2, 0).cpu().numpy(), k=10)
        intersec_map = (1 - im_xdog) * attn_map
        attn_map = intersec_map

    attn_map_soft = np.copy(attn_map)
    attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)

    k = self.num_stages * self.num_paths
    self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False,
                                 p=attn_map_soft.flatten())
    self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T

    self.inds_normalised = np.zeros(self.inds.shape)
    self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
    self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
    self.inds_normalised = self.inds_normalised.tolist()
    return attn_map_soft


def get_path(canvas_width, canvas_height):
    points = []
    strokes_counter = 0
    num_control_points = torch.zeros(args.num_paths, dtype=torch.int32) + (args.control_points_per_seg - 2)
    # p0 = inds_normalised[strokes_counter] if args.attention_init else (random.random(), random.random())
    p0 = (random.random(), random.random())
    points.append(p0)

    for j in range(args.num_paths):
        radius = 0.05
        for k in range(args.control_points_per_seg - 1):
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            points.append(p1)
            p0 = p1
    points = torch.tensor(points).to(device)
    points[:, 0] *= canvas_width
    points[:, 1] *= canvas_height

    path = pydiffvg.Path(num_control_points=num_control_points,
                         points=points,
                         stroke_width=torch.tensor(args.width),
                         is_closed=False)
    strokes_counter += 1
    return path

def load_svg(path_svg):
    svg = os.path.join(path_svg)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        svg)
    return canvas_width, canvas_height, shapes, shape_groups


def init_image(shapes, shape_groups, canvas_width, canvas_height, stage=0):
    optimize_flag = []
    if stage > 0:
        # if multi stages training than add new strokes on existing ones
        # don't optimize on previous strokes
        # optimize_flag = [False for i in range(len(self.shapes))]
        for i in range(args.num_paths):
            # stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            path = get_path(canvas_width, canvas_height)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                             fill_color=None,
                                             stroke_color=torch.tensor(
                                                 [random.random(), random.random(), random.random(), random.random()]))
            shape_groups.append(path_group)
            optimize_flag.append(True)

    else:
        num_paths_exists = 0
        if args.svg != "none":
            canvas_width, canvas_height, shapes, shape_groups = load_svg(args.svg)
            # if you want to add more strokes to existing ones and optimize on all of them
            num_paths_exists = len(shapes)

        for i in range(num_paths_exists, args.num_paths):
            # stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            path = get_path(canvas_width, canvas_height)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                             fill_color=None,
                                             stroke_color=torch.tensor(
                                                 [random.random(), random.random(), random.random(), random.random()]))
            shape_groups.append(path_group)
        optimize_flag = [True for i in range(len(shapes))]

    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg)
    # canvas_width, canvas_height = 256, 256
    render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene( \
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width,  # width
                  canvas_height,  # height
                  2,  # num_samples_x
                  2,  # num_samples_y
                  0,  # seed
                  None,  # bg
                  *scene_args)
    # The output image is in linear RGB space. Do Gamma correction before saving the image.
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1],
                                                         3, device=pydiffvg.get_device()) * (
                   1 - img[:, :, 3:4])
    pydiffvg.imwrite(img.cpu(), './content/svg_img/init.png', gamma=1.0)
    img = img[:, :, :3]
    # Convert img from HWC to NCHW
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2).to(device)  # NHWC -> NCHW
    return img


