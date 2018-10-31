# --------------------------------------------------------
# RON
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and modified by Tao Kong
# --------------------------------------------------------

"""
RON config system.
"""
import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()
__C.MINANCHOR = 32
__C.DIV = 64

# custom
# ---------------------------------
# set train batch size
__C.TRAIN.BATCHSIZE = 5

# set allowed border of anchor
__C.TRAIN.NOBORDER = 0

# set the generation of anchors
__C.GENERATION_ANCHOR_RATIOS = [0.5, 1, 2]  # [1] [0.5, 1, 2] [0.333, 0.5, 1, 2, 3]

# set offset of anchor scale e.g., if scale = 2, another scale = 2*offset
#__C.ANCHOR_SCALE_OFFSET = 1.5

# set supported db used to train and test
__C.SUPPORT_DB_lIST = ['jfa', 'wider', '300w', 'voc']

# set ms-rpn name and scale

# __C.MULTI_SCALE_RPN_NO = ['5', '4']
# __C.TRAIN.MULTI_SCALE_RPN_SCALE = [[8, 16, 32], [2, 4, 8]]

# __C.MULTI_SCALE_RPN_NO = ['5', '4', '3']
# __C.TRAIN.MULTI_SCALE_RPN_SCALE = [[16, 32], [8, 16], [4, 8]]

# __C.MULTI_SCALE_RPN_NO = ['6', '5', '4']
# __C.TRAIN.MULTI_SCALE_RPN_SCALE = [[8, 16], [4, 8], [2, 4]]

# __C.MULTI_SCALE_RPN_NO = ['7', '6', '5']
# __C.TRAIN.MULTI_SCALE_RPN_SCALE = [[8, 16], [4, 8], [2, 4]]

# __C.MULTI_SCALE_RPN_NO = ['5', '4']
# __C.TRAIN.MULTI_SCALE_RPN_SCALE = [[8, 16, 32], [2, 4, 8]]

__C.MULTI_SCALE_RPN_NO = ['5']
__C.TRAIN.MULTI_SCALE_RPN_SCALE = [[1, 2, 4, 8, 16, 32]]

# Iterations between record metrics
__C.TRAIN.METRIC_ITERS = 5000  # 5000 160000
# Metric setting
__C.TRAIN.TestDataSet = ''
__C.TRAIN.TestImgList = ''
__C.TRAIN.TestMetrics = ''
__C.TRAIN.TestPrototxt = ''

# Iterations between record loss
__C.TRAIN.LOSS_ITERS = 1

# enable solverstate
__C.TRAIN.WITH_SOLVERSTATE = 1

# ---------------------------------
__C.TRAIN.ADAPT_SCALE = 0
__C.TRAIN.MIN_MAX_SCALES = [[600, 1000]]
__C.TRAIN.IMAGE_SCALE = 1

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (640,)  # 320, 256
__C.TRAIN.CROPS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
__C.TRAIN.EXPENSION = False
__C.TRAIN.EXPENSION_RAND = True
__C.TRAIN.EXPENSION_SCALE = 2.0
# data augmentation
__C.TRAIN.MOREAUGMENT = False  # False

__C.TRAIN.COLORDISTORATION = False
__C.TRAIN.COLOR_ENHANCE_HI = 1.5
__C.TRAIN.COLOR_ENHANCE_LO = 0.5
__C.TRAIN.GT_OVERLAP = 0.5
# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 4

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5
__C.TRAIN.DET_POSITIVE_OVERLAP = 0.6
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.3
__C.TRAIN.BG_THRESH_LO = 0.001  # rpn:[0, 0.3) ron:(0.001, 0.3)
__C.TRAIN.PROB = 0.03

# additional anchor sample strategy
__C.TRAIN.ALLOW_GT_MAX_OVERLAPS_TO_FG = 1
__C.TRAIN.GT_MAX_OVERLAPS_THRESH = 0.1  # rpn:>0 ron:>0.1

# add OHEM strategy
__C.TRAIN.ALLOW_OHEM_TO_BG = 0

# set batch each module
__C.TRAIN.SET_RPN_EACH_BATCH = 0
__C.TRAIN.RPN_EACH_BATCH = 256

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 10000
# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
__C.TRAIN.USE_PREFETCH = False
# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)



__C.TEST = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (640,)  # (320,)
__C.TEST.PROB = 0.03
__C.TEST.DET_MIN_PROB = 0.01
__C.TEST.BATCH_SIZE = 1
__C.TEST.BOXES_PER_CLASS = 80
__C.TEST.NMS = 0.4
__C.TEST.RON_MIN_SIZE = 10

__C.TEST.ADAPT_SCALE = 0
__C.TEST.MIN_MAX_SCALES = [[600, 1000]]
__C.TEST.IMAGE_SCALE = 1  # only support test batch size = 1
#
# MISC
#
# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG16
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0

# Place outputs under an experiments directory
# __C.EXP_DIR = 'default'
__C.EXP_DIR = 'voc'

def get_output_dir(output_name, subdir='output'):
    """Return the directory where experimental artifacts are placed.
    """
    path = os.path.join(__C.ROOT_DIR, subdir, output_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_dir(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    return dirPath

def get_output_dir_temp(imdb, net):
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is None:
        return path
    else:
        return osp.join(path, net.name)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
