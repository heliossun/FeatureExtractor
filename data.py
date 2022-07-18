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
import skimage


def get_txtNoise(level):
    noise = [delete_random_token,
            replace_random_token,
            random_token_permutation]
    print("noise name: ", noise[level])
    return noise[level]

def noisy(noise_typ,image_path):
    image = skimage.io.imread(image_path)/255.0
    if noise_typ == "gauss":
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,size = image.shape)

      noisy_image = np.clip((image + gauss),0,1)
      image = Image.fromarray((noisy_image*255).astype(np.uint8))

      return image
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy

def noisy2(noise_typ,image_path):
    img = skimage.io.imread(image_path)/255.0
    if noise_typ is not None:
        gimg = skimage.util.random_noise(img,mode = noise_typ)
    skimage.io.imsave("./imgPreprocess/test.png",gimg)
    image = Image.fromarray((skimage*255).astype(np.uint8))
    return image


def get_transform(level):
    # noise = [transforms.RandomAutocontrast(0.5),
    #          transforms.RandomAdjustSharpness(0.5),
    #          transforms.RandomInvert(0.5),
    #          transforms.RandomRotation(degrees=(0, 180)),
    #          transforms.RandomPosterize(bits=2),
    #          transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 10)),
    #          transforms.RandomCrop(128),
    #          transforms.functional.hflip
    #          ]
    #noise = noise[level:level+1]

    noise = [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 10))]
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
