import os
import random
import torchvision
from PIL import Image
import cv2
import numpy as np
import json
import torch
from PIL import Image, ImageFont
from PIL import ImageDraw
import PIL

dataset_path = r"E:\code\synthesis\WordStylist-main\dataset_2"
src_font = "TW-Sung-98_1.ttf"

import torch

def gram_matrix(input_tensor):
    batch_size, channels, height, width = input_tensor.size()
    features = input_tensor.view(batch_size, channels, height * width)
    gram = torch.bmm(features, features.transpose(1, 2))  # ��������˷�
    gram = gram / (channels * height * width)

    return gram


def getFileList(dir, Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist
def latent2image(vae, latent):
    latents = 1 / 0.18215 * latent
    image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)

    return image

def image2latent(vae, images):
    images = vae.encode(images.to(torch.float32)).latent_dist.sample()
    latent = images * 0.18215

    return latent

def draw_single_char(img, canvas_size, char_size):
    width, height = img.size
    factor = width * 1.0 / char_size

    max_height = canvas_size * 2
    if height / factor > max_height:  # too long
        img = img.crop((0, 0, width, int(max_height * factor)))
    if height / factor > char_size + 5:  # CANVAS_SIZE/CHAR_SIZE is a benchmark, height should be less
        factor = height * 1.0 / char_size

    img = img.resize((int(width / factor), int(height / factor)), resample=PIL.Image.LANCZOS)

    bg_img = Image.new("L", (canvas_size, canvas_size), 255)
    offset = ((canvas_size - img.size[0]) // 2, (canvas_size - img.size[1]) // 2)
    bg_img.paste(img, offset)
    return bg_img

def _draw_single_char(font, ch, width, height):
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, fill=0, font=font)
    return img

def get_textsize(font, ch):
    img = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(img)
    char_size = draw.textsize(ch, font=font)
    return char_size

def draw_single_char_by_font(ch):
    font = ImageFont.truetype(src_font, 256)
    width, height = get_textsize(font, ch)
    char_img = _draw_single_char(font, ch, width, height)

    return draw_single_char(char_img, 256, 256)

def load_specific_dict(model, pretrained_model, par):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model)
    if par in list(pretrained_dict.keys())[0]:
        count = len(par) + 1
        pretrained_dict = {k[count:]: v for k, v in pretrained_dict.items() if k[count:] in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    if len(pretrained_dict) > 0:
        model_dict.update(pretrained_dict)
    else:
        return ValueError
    return model_dict

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def crop_whitespace(img):
    img_gray = img.convert("L")
    img_gray = np.array(img_gray)
    ret, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    rect = img.crop((x, y, x + w, y + h))
    return np.array(rect)


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    im = torchvision.transforms.ToPILImage()(grid)
    im.save(path)
    return im


def save_single_images(images, path):
    image = images.squeeze(0)
    white_crop = False
    im = torchvision.transforms.ToPILImage()(image)
    if white_crop == True:
        im = crop_whitespace(im)
        im = Image.fromarray(im)
    else:
        im = im.convert("L")

    im.save(path)
    return im

def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

def get_val_char():
    f = open(os.path.join(dataset_path, "val_char.txt"), 'r', encoding='utf-16')
    val_list = f.read()
    val_list = list(val_list)

    return val_list

def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)