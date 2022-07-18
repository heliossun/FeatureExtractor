# FeatureExtractor
Extract feature map use CLIP Model. Current support COCO & Flickr30k dataset.
Extracted features are used for image-text retrieval.

local reimplement result and result in paper

| Method        |Text Retrieval | | |  Image Retrieval | | |
| :-------------|:-------------  | :------------- | :-------------|:-----| :-------------|:------------- |
|               |R@1 |R@5|R@10   | R@1|R@5|R@10|       
| paper(COCO)   |88.2|98.7|99.4|68.7|90.6|95.2|
| ours(COCO)   |88.8|98.0|99.3|70.76|89.92|93.68|
| paper(F30k)   |58.4|81.5|88.1|37.8|62.4|72.2|
|ours(F30k)    |57.3|80.4|87.66|36.29|60.72|70.66|

MSCOCO performance is reported on the 5k val set.
Flickr30k performance is reported on the 1k val set.
Use pretrained model: openai/clip-vit-large-patch14-336
