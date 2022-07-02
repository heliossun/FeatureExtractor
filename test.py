# get model
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from evaluation_5N import i2t, t2i
from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file

model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14-336')

# get dataset
import json
with open("/data/Guohao/datasets/coco/annotations/captions_val2017.json",'r') as load_f:
    coco = json.load(load_f)
coco_pair = {}
for i in range(len(coco['images'])):
    coco_pair[coco['images'][i]['id']] = [coco['images'][i]['coco_url']]
for i in range(len(coco['annotations'])):
    coco_pair[coco['annotations'][i]['image_id']].append(coco['annotations'][i]['caption'])

# get mask
import torch

# visual mask 24 layers, 16 heads
visual_mask = torch.ones([24, 1, 16, 1, 1])
# text mask 12 layers, 12 heads
text_mask = torch.ones([12, 1, 12*5, 1, 1])

# pick pruning index: random
prune_type = 'rand'
V_mask = 'use'
L_mask = 'nouse'
pr = 0.1
visual_list = [i for i in range(visual_mask.size()[2])]
text_list = [i for i in range(text_mask.size()[2])]
import random

for i in range(visual_mask.size()[0]):
    random.shuffle(visual_list)
    sub_list = visual_list[:int(len(visual_list)*pr)]
    for j in sub_list:
        visual_mask[i,:,j,:,:] = 0

for i in range(text_mask.size()[0]):
    random.shuffle(text_list)
    sub_list = text_list[:int(len(text_list)*pr)]
    for j in sub_list:
        text_mask[i,:,j,:,:] = 0

# get features
from pdb import set_trace as st
import numpy as np
import time

img_feats = None
txt_feats = None

model.cuda()

# num_sample = len(coco_pari.keys())
num_sample = 10

for i in range(num_sample):
    id_ = list(coco_pair.keys())[i]
    
    img = Image.open(requests.get(coco_pair[id_][0], stream=True).raw)
    txt = coco_pair[id_][1:][:5]
    # st()

    inputs = processor(text=txt, images=img, return_tensors="pt", padding=True)
    # if V_mask == 'use':
    #     inputs['head_mask_visual'] = visual_mask
    # if L_mask == 'use':
    #     inputs['head_mask_text'] = text_mask
    inputs = {k:v.to('cuda') for k,v in inputs.items()}
    # st()
    # tik = time.time()
    outputs = model(**inputs)
    # tok = time.time()
    # print(tok-tik)
    # img
    if img_feats is None:
        img_feats = np.zeros((num_sample, outputs.image_embeds.size(1)))
        txt_feats = np.zeros((num_sample*5, outputs.image_embeds.size(1)))
    img_feats[i]=outputs.image_embeds.detach().cpu().numpy()[0]
    # txt only count 5, some with 6
    for j in range(5):
        txt_feats[i*5+j]=outputs.text_embeds.detach().cpu().numpy()[j]
    # if i % 200 == 0:
    #     print(i)
    #     print(np.array(img_feats).shape)
    #     print(np.array(txt_feats).shape)

(r1, r5, r10, medr, meanr) = i2t(img_feats, txt_feats, measure="cosine")
(r1i, r5i, r10i, medri, meanr) = t2i(img_feats, txt_feats, measure="cosine")
LOGGER.info(
    f"========================= validation ===========================\n"
    f"image retrieval R1: {r1i:.2f},\n"
    f"image retrieval R5: {r5i:.2f},\n"
    f"image retrieval R10: {r10i:.2f}\n"
    f"text retrieval R1: {r1:.2f},\n"
    f"text retrieval R5: {r5:.2f},\n"
    f"text retrieval R10: {r10:.2f}")
LOGGER.info("=========================================================")
# save features
np.save('hg_pruned_features/' + str(num_sample) + 'img'  + '.npy', img_feats)
np.save('hg_pruned_features/' + str(num_sample) + 'txt'  + '.npy', txt_feats)
