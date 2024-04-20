import argparse
import os
import cv2
import numpy as np
import copy
from PIL import Image, ImageEnhance
import PIL
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm
import fastremap
from cellpose.flow import masks_to_flows

def labels_to_flows(instances, niter=None, device="cuda"):
    if instances.ndim < 3:
        instances = instances[np.newaxis, :, :] 

    # flows need to be recomputed
    instances = fastremap.renumber(instances, in_place=True)[0]
    veci = masks_to_flows(instances[0].astype(int), device=device, niter=niter)

    # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
    flows = np.concatenate((instances > 0.5, veci),
                           axis=0).astype(np.float32)
    return flows


class EMPSDataset(Dataset):
    def __init__(
        self, data_dir, data_mod='train', im_size=(512, 512), 
        transform=True, device='cuda', labeled_data=True, 
        extra_data=False, weak_data=False,
        expand_image=False, inverse_image=False
    ):
        self.data_dir = data_dir
        self.im_size = im_size
        self.transform = transform
        self.device = device
        self.base_dir = os.path.join(data_dir, data_mod)
        self.expand_image = expand_image
        self.inverse_image = inverse_image
        self.image_fns = []

        if labeled_data:
            image_fns = os.listdir(os.path.join(self.base_dir, 'images'))
            for img in image_fns:
                if img.endswith('.png'):
                    labels = np.array(Image.open(os.path.join(self.base_dir, "segmaps", img)))
                    self.image_fns.append((self.base_dir, img, True))
            print('[INFO] add real data:', len(self.image_fns))

        if extra_data:
            print('[Loading] extra data ...')
            num = len(self.image_fns)
            extra_data_base = os.path.join(data_dir, 'extra_data')
            for exp in os.listdir(extra_data_base):
                if os.path.isdir(os.path.join(extra_data_base, exp)):
                    for img in tqdm(os.listdir(os.path.join(extra_data_base, exp, 'images')), desc=exp):
                        labels = np.array(Image.open(os.path.join(extra_data_base, exp, "segmaps", img)))
                        # if labels.max() < 600:
                        #     self.image_fns.append((os.path.join(extra_data_base, exp), img, True))
                        self.image_fns.append((os.path.join(extra_data_base, exp), img, True))
            print('[INFO] add extra data:', len(self.image_fns) - num)
            
        if weak_data:
            num = len(self.image_fns)
            extra_data_base = os.path.join(data_dir, 'weak_data')
            for img in os.listdir(os.path.join(extra_data_base, 'images')):
                self.image_fns.append((extra_data_base, img, False))
            print('[INFO] add weak data:', len(self.image_fns) - num)

        self.colour_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.3, hue=0.3)
        self.crop = transforms.RandomCrop((self.im_size[0]//2, self.im_size[1]//2))
        self.hor_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.vert_flip = transforms.RandomVerticalFlip(p=1.0)

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.image_fns[idx][0], 'images', self.image_fns[idx][1])
        )
        image = image.resize(self.im_size, resample=Image.BICUBIC).convert("RGB")
        
        if self.image_fns[idx][2]:
            instances = Image.open(
                os.path.join(self.image_fns[idx][0], 'segmaps', self.image_fns[idx][1])
            ).resize(self.im_size, resample=Image.NEAREST)
            labels = Image.fromarray((np.array(instances) > 0).astype(np.uint8))
        else:
            # 需要检查一下
            masks = cv2.imread(os.path.join(self.image_fns[idx][0], 'weak_labels', self.image_fns[idx][1]), 0)
            W = int(masks.shape[1] / 2)
            instances = Image.fromarray((masks[:, :W] > 0).astype(np.uint8)).resize(self.im_size, resample=Image.NEAREST)
            labels = Image.fromarray((masks[:, W:] > 0).astype(np.uint8)).resize(self.im_size, resample=Image.NEAREST)
            
        is_full_labeled = self.image_fns[idx][2]

        if self.transform:

            # reduce quality
            raw = copy.deepcopy(image)
            image = self.random_reduce_quality(image)
            # if np.random.uniform() < 0.5:
            #     image = image.filter(PIL.ImageFilter.SHARPEN)
            # if np.random.uniform() < 0.5:
            #     image = image.filter(PIL.ImageFilter.SMOOTH)
            # if np.random.uniform() < 0.5:
            #     image = PIL.ImageOps.equalize(image, mask=None)
            # if np.random.uniform() < 0.5:
            #     brightEnhancer = ImageEnhance.Brightness(image)
            #     image = brightEnhancer.enhance(np.random.uniform() * 0.7 + 0.5)
            # hor-ver flip
            image, instances, labels = self.horizontal_flip(image, instances, labels)
            image, instances, labels = self.vertical_flip(image, instances, labels)

            # rotate
            image, instances, labels = self.random_rotation(image, instances, labels)

            # colour jitter
            # image = self.colour_jitter(image)
            if self.inverse_image and np.random.uniform() < 0.5:
                image = PIL.ImageOps.invert(image)

            # random crop
            image, instances, labels = self.random_crop(image, instances, labels)
            
            # random expand
            if self.expand_image:
                image, instances, labels = self.expand(image, instances, labels)

            # tile
            image, instances, labels = self.tile(image, instances, labels)

        # scale
        image = np.array(image) / 255.0

        image = torch.FloatTensor(image).permute(2, 0, 1)
        
        if is_full_labeled:
            flows = labels_to_flows(np.array(instances), device=torch.device("cuda"))
            flows = torch.tensor(flows)
        else:
            flows = torch.zeros_like(image)
        
        instances = torch.LongTensor(np.array(instances))
        labels = torch.ByteTensor(np.array(labels))

        return image, instances, labels, flows, is_full_labeled

    def __len__(self):
        return len(self.image_fns)

    def horizontal_flip(self, image, instances, labels, p=0.5):
        if np.random.uniform() < p:
            image = self.hor_flip(image)
            instances = self.hor_flip(instances)
            labels = self.hor_flip(labels)
        return image, instances, labels

    def expand(self, image, instances, labels, p=0.5):
        if np.random.uniform() < p:
            scale_factor = np.random.choice([1/2, 1/2, 1/2, 1/2, 1/4, 1/4, 1/8])
            tgt_size = (int(self.im_size[0]*scale_factor), int(self.im_size[1]*scale_factor))
            pad = (int((self.im_size[0] - tgt_size[0])/2), int((self.im_size[1] - tgt_size[1])/2))
            image = image.resize(tgt_size, resample=Image.BICUBIC)
            instances = instances.resize(tgt_size, resample=Image.BICUBIC)
            labels = labels.resize(tgt_size, resample=Image.BICUBIC)
            image = transforms.functional.pad(image, pad)
            instances = transforms.functional.pad(instances, pad)
            labels = transforms.functional.pad(labels, pad)
        return image, instances, labels

    def vertical_flip(self, image, instances, labels, p=0.5):
        if np.random.uniform() < p:
            image = self.vert_flip(image)
            instances = self.vert_flip(instances)
            labels = self.vert_flip(labels)
        return image, instances, labels

    def random_rotation(self, image, instances, labels):
        random_number = np.random.uniform()
        if random_number < 0.25:
            image = image.rotate(90)
            instances = instances.rotate(90)
            labels = labels.rotate(90)
        elif (random_number >= 0.25) & (random_number < 0.5):
            image = image.rotate(180)
            instances = instances.rotate(180)
            labels = labels.rotate(180)
        elif (random_number >= 0.5) & (random_number < 0.75):
            image = image.rotate(270)
            instances = instances.rotate(270)
            labels = labels.rotate(270)
        return image, instances, labels

    def random_crop(self, image, instances, labels):
        if np.random.uniform() <= 0.333:
            i, j, h, w = self.crop.get_params(image, self.crop.size)
            image_cropped = transforms.functional.crop(image, i, j, h, w).resize(self.im_size, resample=Image.BICUBIC)
            instances_cropped = transforms.functional.crop(instances, i, j, h, w).resize(self.im_size, resample=Image.NEAREST)
            labels_cropped = transforms.functional.crop(labels, i, j, h, w).resize(self.im_size, resample=Image.NEAREST)
            # recursively random crop until n instances > 0 (not a problem in almost all cases.)
            if len(np.unique(np.array(labels_cropped))) == 1:
                image_cropped, instances_cropped, labels_cropped = self.random_crop(image, instances, labels)
            return image_cropped, instances_cropped, labels_cropped
        else:
            return image, instances, labels

    def random_reduce_quality(self, image):
        scale_factor = np.random.choice([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1/2, 1/2, 1/2, 1/2, 1/4, 1/4])
        image = image.resize((int(self.im_size[0]*scale_factor), int(self.im_size[1]*scale_factor)))
        image = image.resize(self.im_size, resample=Image.NEAREST)
        return image

    def tile(self, image, instances, labels, p=0.05):
        if np.random.uniform() <= p:
            tile_size = np.array(self.im_size) // 2
            image = image.resize(tile_size, resample=Image.BICUBIC)
            instances = instances.resize(tile_size, resample=Image.NEAREST)
            labels = labels.resize(tile_size, resample=Image.NEAREST)

            image = np.tile(np.array(image), (2, 2, 1))
            instances = np.tile(np.array(instances), (2, 2))
            labels = np.tile(np.array(labels), (2, 2))

        return image, instances, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on EMPS dataset.')
    parser.add_argument('--data-dir', metavar='data_dir', type=str, help='Directory which contains the data.')
    parser.add_argument('--device', metavar='device', type=str, default='cuda', help='device to train on (cuda or cpu)')
    parser.add_argument('--im-size', metavar='im_size', type=tuple, default=(512, 512), help='Image size to load for training.')
    parser.add_argument('--labeled_data', action='store_true', help="increase h/w gradually for smoother texture")
    parser.add_argument('--extra_data', action='store_true', help="increase h/w gradually for smoother texture")
    parser.add_argument('--weak_data', action='store_true', help="increase h/w gradually for smoother texture")
    namespace = parser.parse_args()
    train_dataset = EMPSDataset(namespace.data_dir, 
                                data_mod='train',
                                im_size=namespace.im_size, 
                                device=namespace.device,
                                labeled_data=namespace.labeled_data,
                                extra_data=namespace.extra_data,
                                weak_data=namespace.weak_data)

    test_dataset = EMPSDataset(namespace.data_dir, 
                                data_mod='eval',
                                im_size=namespace.im_size, 
                                device=namespace.device,
                                labeled_data=True)
    
    train_dataset.__getitem__(-1)
