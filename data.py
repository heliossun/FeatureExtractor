import numpy as np
from torch import nn
import torch
import os
import torch.utils.data as data
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets.coco import CocoCaptions
from torchvision.datasets.flickr import Flickr30k, Flickr8k
from PIL import Image
from utils.txt_noise import *
import json

def get_txtNoise(level):
    noise = [delete_random_token,
            replace_random_token,
            random_token_permutation]
    print("noise name: ", noise[level])
    return noise[level]


def get_transform(level):
    noise = [transforms.RandomAutocontrast(0.5),
             transforms.RandomAdjustSharpness(0.5),
             transforms.RandomInvert(0.5),
             transforms.RandomRotation(degrees=(0, 180)),
             transforms.RandomPosterize(bits=2),
             transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 10)),
             transforms.RandomCrop(128),
             transforms.functional.hflip
             ]
    #noise = [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 10))]

    noise = noise[level:level+1]
    transform = transforms.Compose(noise)
    print(noise)


    return transform

class CocoDataset():
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, transform=None, ids=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            transform: transformer for image.
        """
        self.coco = COCO(json)
        self.root = root
        # if ids provided by get_paths, use split-specific ids
        if ids is None:
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = ids
        self.transform = transform

    def items(self):
        coco_pair = {}
        for i in range(len(self.ids)):
            caption, img_id, path = self.get_raw_item(i)
            if img_id not in coco_pair.keys():
                coco_pair[img_id]=[path,[]]
            coco_pair[img_id][1].append(caption)
        print("# of different images: ",len(coco_pair))
        return coco_pair

    def get_raw_item(self, index):

        ann_id = self.ids[index]
        #print(self.coco.anns[ann_id])
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        #print(caption)
        #print(img_id)
        #image = Image.open(os.path.join(self.root, path)).convert('RGB')
        return caption, img_id, path



class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, anno, transform=None):
        self.root = root
        self.transform = transform
        self.ann = json.load(open(anno, 'r'))


    def items(self):
        """This function returns a tuple that is further passed to collate_fn
        """
        root = self.root
        flickr_pair = {}
        for i in range(len(self.ann)):
            ann = self.ann[i]
            if i not in flickr_pair.keys():
                flickr_pair[i]=[ann['image'],ann['caption']]
        print("# of different images: ",len(flickr_pair))
        return flickr_pair

    def __len__(self):
        return len(self.ids)

def get_data(data_name, root, json, batch_size):

    if data_name =='coco' :
        # COCO custom dataset
        dataset = CocoCaptions(root = root,
                               annFile = json,
                               transform = None)
        dataset = DataLoader(dataset, batch_size = batch_size)
    elif data_name == 'coco_' :
        dataset = CocoDataset(root = root,
                               json = json,
                               transform = None).items()
    elif 'f30k' in data_name:
        dataset = FlickrDataset(root=root,
                            anno = json,
                            transform=None).items()
    elif 'f8k' in data_name:
        dataset = Flickr8k(root=root,
                            ann_file=json,
                            transform=None)


    return dataset
