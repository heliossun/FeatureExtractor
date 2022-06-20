from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import data
from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
import argparse
from utils.misc import parse_with_config


def validate(val_data, model, processor, opts):

    images, texts = val_data
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)


def main(opts):
    LOGGER.info(f"Loading Val Dataset {opts.root}, "
                f"{opts.anno}")

    val_data = data.get_data(opts.dataname, opts.root, opts.anno, opts.batch_size)
    model = CLIPModel.from_pretrained(opts.model)
    processor = CLIPProcessor.from_pretrained(opts.model)
    validate(val_data, model, processor, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='coco',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)
    main(args)
