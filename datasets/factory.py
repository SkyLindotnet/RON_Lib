# --------------------------------------------------------
# RON
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and modified by Tao Kong
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.pascal_voc
import datasets.coco
import numpy as np
from datasets.general_voc import general_voc

import os
from fast_rcnn.config import cfg

# custom
# __sets['voc_wider'] = (lambda name='voc_wider', split='train': general_voc(name, split, '../data/DB/object/voc_wider'))

debugTrainPath = os.path.join(cfg.ROOT_DIR, 'data/VOCdevkit2007')
__sets['voc_2007_trainval_debug'] = (lambda name='trainval', split='2007': datasets.pascal_voc(name, split, debugTrainPath))
debugTestPath = os.path.join(cfg.ROOT_DIR, 'data/VOCdevkit2007')
__sets['voc_2007_test_debug'] = (lambda name='test', split='2007': datasets.pascal_voc(name, split, debugTestPath))

for split in ['train', 'val']:
    name = 'wider_{}'.format(split)
    path = os.path.join(cfg.ROOT_DIR, 'data/DB/object/voc_wider')
    __sets[name] = (lambda name=name, split=split: general_voc(name, split, path))

    # __sets['jfa'] = (lambda name='jfa_{}'.format(split), split=split: general_face(name, split, '../data/DB/face/Face_plus'))
    #
    # __sets['300w'] = (lambda name='300w_{}'.format(split), split=split: general_face(name, split, '../data/DB/face/300-w_face'))
    #
    # __sets['morph'] = (lambda name='morph_{}'.format(split), split=split: general_face(name, split, '../data/DB/face/Morph'))


# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

for year in ['2014']:
    for split in ['train', 'val']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: 
        		datasets.coco(split, year))
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                        datasets.coco(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
