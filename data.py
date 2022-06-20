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

def get_transform():
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform

class CocoDataset():
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, transform=None, ids=None, size=1000):
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
        self.size = size
    def items(self):
        """This function returns a tuple that is further passed to collate_fn
        """
        captions = []
        images = []
        for i in range(self.size):
            caption, image = self.get_raw_item(i)
            captions.append(caption)
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)


        return images, captions

    def get_raw_item(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        return caption, image



class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)

def get_data(data_name, root, json, batch_size):
    transform = get_transform()
    if 'coco' in data_name:
        # COCO custom dataset
        dataset = CocoDataset(root = root,
                               json = json,
                               transform = transform).items()
    elif 'f30k' in data_name:
        dataset = Flickr30k(root=root,
                            ann_file = json,
                            transform=None)
    elif 'f8k' in data_name:
        dataset = Flickr8k(root=root,
                            ann_file=json,
                            transform=None)


    return dataset
