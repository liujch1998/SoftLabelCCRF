# SoftLabelCCRF

This repository contains implementation of [*Phrase Grounding by Soft-Label Chain Conditional Random Field*](https://arxiv.org/abs/1909.00301) in EMNLP-IJCNLP 2019.

## Setup

This repo is tested on Python 3.6 and PyTorch 1.0.0

Install dependencies: numpy, cv2, progressbar, allennlp

Download and extract the [Flickr30kEntities dataset](http://bryanplummer.com/Flickr30kEntities/)

Clone this repo. In this repo, 
* Create directory `feats` and `cache`
* Create symlink to `[path-to-Flickr30kEntities-dataset]`

## Image Feature Extraction and Caching

Clone the [Bottom-Up Attention](https://github.com/peteanderson80/bottom-up-attention) repo

Create directory `[path-to-bottom-up-attention-repo]/data/flickr30k`. In this directory, 
* Create symlink to `[path-to-Flickr30kEntities-dataset]`
* Create symlink to `[path-to-SoftLabelCCRF-repo]/split`
* Create symlink to `[path-to-SoftLabelCCRF-repo]/feats`

Replace `[path-to-bottom-up-attention-repo]/tools/generate_tsv.py` with the one provided by us in `[path-to-SoftLabelCCRF-repo]/tools/generate_tsv.py`

Under `[path-to-bottom-up-attention-repo]`, run something like
```
python ./tools/generate_tsv.py \
    --gpu 1,2,3,4,5,6,7 \
    --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
    --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml \
    --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel \
    --split flickr30k \
    --out data/flickr30k/feats/feats.tsv
```
Region proposals and features will appear in `[path-to-SoftLabelCCRF-repo]/feats`

Under `[path-to-SoftLabelCCRF-repo]`, run something like
```
python3.6 SoftLabelCCRF/cache/cache.py 1-7 29783 1000 1000
```
Per-image caches will appear in `[path-to-SoftLabelCCRF-repo]/cache`

## Training and Testing

```
CUDA_VISIBLE_DEVICES=1 python3.6 SoftLabelCCRF/run.py \
    --output_dir [output_dir] \
    --do_train \
    --do_test
```
Additionally, you may use
* `--kld` to switch from Hard-Label to Soft-Label training
* `--crf` to enable CRF layer. Be sure to specify `--decode [viterbi|smoothing]`
* `--tran_context [none|m|mlr|mlrg]` to add transition score context
* `--no_box_regression` to disable bounding box regression. Be sure to set `--gamma 0.0`
* `--visualize` to visualize grounding results

Other flags can be found in the script

## Trained Models

| Handle | Description | Flags | Size |
| ------ | ----------- | ----- | ---- |
| [HL](https://drive.google.com/file/d/1P_xs5AYRIGNOK9IiU13HxKdGeex--Kg2/view?usp=sharing) | Hard-Label Model | N/A | 477.6 MB |
| [HL-CCRF](https://drive.google.com/file/d/1Je16eiTcns6firDoewZE46ewlMdiHXxo/view?usp=sharing) | Hard-Label Chain CRF Model | `--crf --decode viterbi` | 478.1 MB |
| [SL](https://drive.google.com/file/d/1IETz6vwDbN6jZCpDYXEiUfUwmyONz1Wh/view?usp=sharing) | Soft-Label Model | `--kld` | 477.6 MB |
| [SL-CCRF](https://drive.google.com/file/d/1GKdsMhR3W_-oPEtUKKWsPvIIVqmUEj2f/view?usp=sharing) | Soft-Label Chain CRF Model | `--kld --crf --decode viterbi` | 478.1 MB |
| [SL-CCRF-M](https://drive.google.com/file/d/1hUvP391mwIv276ji3fwy7n4DuseEUqgu/view?usp=sharing) | SL-CCRF, M | `--kld --crf --decode viterbi --tran_context m` | 478.2 MB |
| [SL-CCRF-MLR](https://drive.google.com/file/d/17gKhHOzhcJ_zaQlI1dd6U78N23ndzihM/view?usp=sharing) | SL-CCRF, MLR | `--kld --crf --decode viterbi --tran_context mlr` | 478.6 MB |
| [SL-CCRF-MLRG](https://drive.google.com/file/d/1PdMg_SrrCzcjkIxYgkgyrE81BrzBzc3a/view?usp=sharing) | SL-CCRF, MLRG | `--kld --crf --decode viterbi --tran_context mlrg` | 478.8 MB |
| [SL-CCRF-M-NBR](https://drive.google.com/file/d/1BTEMugUhuhMSWBkifbtsettH-EtPhQY5/view?usp=sharing) | SL-CCRF, M, no box regression | `--kld --crf --decode viterbi --tran_context m --no_box_regression --gamma 0.0` | 478.2 MB |

## Citation

If you find this repo useful, please consider citing the following work
```
@inproceedings{liu-hockenmaier-2019-phrase,
    title = "Phrase Grounding by Soft-Label Chain Conditional Random Field",
    author = "Liu, Jiacheng  and
      Hockenmaier, Julia",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1515",
    pages = "5115--5125",
    abstract = "The phrase grounding task aims to ground each entity mention in a given caption of an image to a corresponding region in that image. Although there are clear dependencies between how different mentions of the same caption should be grounded, previous structured prediction methods that aim to capture such dependencies need to resort to approximate inference or non-differentiable losses. In this paper, we formulate phrase grounding as a sequence labeling task where we treat candidate regions as potential labels, and use neural chain Conditional Random Fields (CRFs) to model dependencies among regions for adjacent mentions. In contrast to standard sequence labeling tasks, the phrase grounding task is defined such that there may be multiple correct candidate regions. To address this multiplicity of gold labels, we define so-called Soft-Label Chain CRFs, and present an algorithm that enables convenient end-to-end training. Our method establishes a new state-of-the-art on phrase grounding on the Flickr30k Entities dataset. Analysis shows that our model benefits both from the entity dependencies captured by the CRF and from the soft-label training regime. Our code is available at {\textbackslash}url{github.com/liujch1998/SoftLabelCCRF}",
}
```

