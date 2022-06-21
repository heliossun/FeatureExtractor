from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import data
from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
import argparse
from utils.misc import parse_with_config
from evaluation_5N import i2t, t2i
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # change
def encoder1(valid_dataloader,model, processor,opts):
    img_embs = np.zeros((len(valid_dataloader.dataset), 512))
    cap_embs = np.zeros((len(valid_dataloader.dataset), 512))
    for i, batch in enumerate(valid_dataloader):
        images, texts = batch

        if opts.single_caption:
            texts = [texts[0][0]]
        else:
            texts = [txt[0] for txt in texts]
        images = torch.squeeze(images,0)
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        txt_embeds = outputs.text_embeds
        img_embeds = outputs.image_embeds

        img_embs[i]=img_embeds.detach().cpu().numpy().copy()
        cap_embs[i]=txt_embeds.detach().cpu().numpy().copy()
        del images, texts

    return img_embs, cap_embs
def encoder(data,model, processor,opts):
    #option 1
    img_embs = None
    cap_embs = None
    img_ids = list(data.keys())
    for i in range(opts.data_size):
        image_pth = data[img_ids[i]][0]
        txts = data[img_ids[i]][1]
        image = Image.open(os.path.join(opts.root, image_pth)).convert('RGB')
        inputs = processor(text=txts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        outputs = model(**inputs)
        txt_embeds = outputs.text_embeds
        img_embeds = outputs.image_embeds
        if img_embs is None:
            img_embs = np.zeros((opts.data_size, img_embeds.size(1)))
            cap_embs = np.zeros((opts.data_size*5, txt_embeds.size(1)))
        img_embs[i]=img_embeds.detach().cpu().numpy()
        for j in range(5):
            cap_embs[5*i+j]=txt_embeds[j].detach().cpu().numpy()


    #option 2
    # images, texts = valid_dataloader
    # inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    # inputs = {k: v.to('cuda') for k, v in inputs.items()}
    # outputs = model(**inputs)
    # txt_embeds = outputs.text_embeds
    # img_embeds = outputs.image_embeds
    #
    # img_embs=img_embeds.detach().cpu().numpy().copy()
    # cap_embs=txt_embeds.detach().cpu().numpy().copy()
    #print(img_embs)
    return img_embs, cap_embs

def validate(valid_dataloader, model, processor, opts):


    img_embs, cap_embs = encoder(valid_dataloader,model, processor,opts)

    #print("img embd size:",img_embs.shape) (5000,512)
    #print("txt embd size:",cap_embs.shape) (5000,512)
    LOGGER.info("========compute ranking score==========")
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure=opts.measure)
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, measure=opts.measure)
    LOGGER.info(
        f"========================= validation ===========================\n"
        f"image retrieval R1: {r1i:.2f},\n"
        f"image retrieval R5: {r5i:.2f},\n"
        f"image retrieval R10: {r10i:.2f}\n"
        f"text retrieval R1: {r1:.2f},\n"
        f"text retrieval R5: {r5:.2f},\n"
        f"text retrieval R10: {r10:.2f}")
    LOGGER.info("=========================================================")
def main(opts):
    LOGGER.info(f"Loading Val Dataset {opts.root}, "
                f"{opts.anno}")

    val_data = data.get_data(opts.dataname, opts.root, opts.anno, opts.batch_size)
    model = CLIPModel.from_pretrained(opts.model).cuda()
    processor = CLIPProcessor.from_pretrained(opts.model)
    validate(val_data, model, processor, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='coco',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)
    main(args)
