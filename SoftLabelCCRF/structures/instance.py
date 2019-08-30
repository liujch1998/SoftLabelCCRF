import os
import xml.etree.ElementTree as ET

import torch
import cv2

from structures.caption import Caption
from structures.region import Region
from utils.vision import merge_bboxes, iou, collect_region_spatial_feats

class Instance:

    def __init__ (self, token, bboxes, regions_visual_feats):
        self.id = token
        self.image_path = 'Flickr30kEntities/Images/all/' + self.id + '.jpg'
        self.caption_path = 'Flickr30kEntities/Sentences/' + self.id + '.txt'
        self.annotation_path = 'Flickr30kEntities/Annotations/' + self.id + '.xml'

        self.captions = []  # [Caption]
        self.image_size = ()  # (width, height, depth)
        self.load_captions()
        self.load_annotations()

        self.regions = []  # [Region] 'candidate regions from region proposal'
        self.image_ = None  # (1, d_image) 'features of the entire image'
#        bboxes, regions_visual_feats, objectnesses, self.image_ = vision_model(cv2.imread(self.image_path))
        self.regions = [Region(bbox, torch.cat((region_visual_feats, collect_region_spatial_feats(bbox, self.image_size)), dim=1)) for bbox, region_visual_feats in zip(bboxes, regions_visual_feats)]

    def load_captions (self):
        # load annotated captions from file
        with open(self.caption_path, 'r') as f:
            lines = f.readlines()
            captions_ann = [line.strip() for line in lines]
            for index, ann in enumerate(captions_ann):
                caption = Caption(index, ann)
                self.captions.append(caption)

    def load_annotations (self):
        # load annotations from file
        tree = ET.parse(self.annotation_path)
        root = tree.getroot()

        # setup image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)
        self.image_size = (width, height, depth)

        # setup mention bbox
        for object in root.findall('object'):
            id = object.find('name').text
            box = object.find('bndbox')
            if box is None:
                continue
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text) + 1
            ymax = float(box.find('ymax').text) + 1
            bbox = [xmin, ymin, xmax, ymax]
            for caption in self.captions:
                for mention in caption.mentions:
                    if mention.id == id:
                        mention.bbox = merge_bboxes(mention.bbox, bbox)

        # remove mentions without bbox
        for caption in self.captions:
            caption.mentions = [mention for mention in caption.mentions if mention.bbox is not None]

    def visualize_prediction (self, output_dir, split):
        for caption in self.captions:
            image = cv2.imread(self.image_path)
            for mention in caption.mentions:
                color = (0,255,0,0) if iou(mention.bbox, mention.bbox_pred) >= 0.5 else (0,0,255,0)
                [xmin, ymin, xmax, ymax] = mention.bbox_pred
                xmin, ymin, xmax, ymax = int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
                cv2.putText(image, mention.raw, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(output_dir, 'visualize', split, '%s %s.jpg' % (self.id, caption.raw[:200])), image)

