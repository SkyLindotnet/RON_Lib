# --------------------------------------------------------
# RON
# Licensed under The MIT License [see LICENSE for details]
# Written by Tao Kong, 2016-11-22
# --------------------------------------------------------
from fast_rcnn.config import cfg, get_output_dir, get_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.bbox_transform import clip_boxes, filter_boxes
import cPickle
import heapq
import shutil
from utils.blob import im_list_to_blob
import os
import matplotlib.pyplot as plt
from datasets.voc_eval import voc_ap

def _get_image_blob(ims, target_size):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_infos(ndarray): a data blob holding input size pyramid
    """
    processed_ims = []
    for im in ims:
        im = im.astype(np.float32, copy=False)
        im = im - cfg.PIXEL_MEANS
        im_shape = im.shape[0:2]
        im = cv2.resize(im, None, None, fx=float(target_size) / im_shape[1], \
                        fy=float(target_size) / im_shape[0], interpolation=cv2.INTER_LINEAR)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob


def _get_image_blob_adapt(ims, min_max_size):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_infos(ndarray): a data blob holding input size pyramid
    """
    processed_ims = []
    for im in ims:
        im = im.astype(np.float32, copy=False)
        im = im - cfg.PIXEL_MEANS
        im_shape = im.shape[0:2]
        im_size_min = np.min(im_shape)
        im_size_max = np.max(im_shape)
        im_scale = float(min_max_size[0]) / float(im_size_min)
        if np.round(im_scale * im_size_max) > min_max_size[1]:
            im_scale = float(min_max_size[1]) / float(im_size_max)

        im = cv2.resize(im, None, None, fx=im_scale, \
                        fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims.append(im)
        cfg.TEST.IMAGE_SCALE = im_scale

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob


def _get_blobs(ims, target_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data': None}
    blobs['data'] = _get_image_blob(ims, target_size)

    return blobs

def _get_blobs_adapt(ims, min_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data': None}
    blobs['data'] = _get_image_blob_adapt(ims, min_max_size)

    return blobs

def im_detect_ron(net, ims):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    if cfg.TEST.ADAPT_SCALE:
        blobs = _get_blobs_adapt(ims, min_max_size=cfg.TEST.MIN_MAX_SCALES[0])
    else:
        blobs = _get_blobs(ims, target_size=cfg.TEST.SCALES[0])

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False))

    pred_boxes7 = blobs_out['rois7']
    scores7 = blobs_out['scores7']

    pred_boxes6 = blobs_out['rois6']
    scores6 = blobs_out['scores6']

    pred_boxes5 = blobs_out['rois5']
    scores5 = blobs_out['scores5']

    pred_boxes4 = blobs_out['rois4']
    scores4 = blobs_out['scores4']

    pred_boxes = np.zeros((cfg.TEST.BATCH_SIZE, 0, 4), dtype=np.float32)
    scores = np.zeros((cfg.TEST.BATCH_SIZE, 0, scores7.shape[-1]), dtype=np.float32)

    scores = np.concatenate((scores, scores7), axis=1)
    scores = np.concatenate((scores, scores6), axis=1)
    scores = np.concatenate((scores, scores5), axis=1)
    scores = np.concatenate((scores, scores4), axis=1)

    pred_boxes = np.concatenate((pred_boxes, pred_boxes7), axis=1)
    pred_boxes = np.concatenate((pred_boxes, pred_boxes6), axis=1)
    pred_boxes = np.concatenate((pred_boxes, pred_boxes5), axis=1)
    pred_boxes = np.concatenate((pred_boxes, pred_boxes4), axis=1)

    return scores, pred_boxes


def im_detect(net, ims):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    if cfg.TEST.ADAPT_SCALE:
        blobs = _get_blobs_adapt(ims, min_max_size=cfg.TEST.MIN_MAX_SCALES[0])
    else:
        blobs = _get_blobs(ims, target_size=cfg.TEST.SCALES[0])

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False))

    pred_boxes = np.zeros((cfg.TEST.BATCH_SIZE, 0, 4), dtype=np.float32)
    scores = np.zeros((cfg.TEST.BATCH_SIZE, 0, 1), dtype=np.float32)  # 2 class

    for rpn_no in cfg.MULTI_SCALE_RPN_NO:
        pred_boxes = np.concatenate((pred_boxes, blobs_out['rois_%s' % rpn_no]), axis=1)
        scores = np.concatenate((scores, blobs_out['scores_%s' % rpn_no]), axis=1)

    return scores, pred_boxes


def vis_detections(im, class_name, dets, thresh=0.5):
    """Visual debugging of detections."""

    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(5, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]

        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
            )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()


def record_detections(filePath, im_name, class_name, dets, thresh=0.5):

    resultFile = open(filePath, 'a')
    # save result
    resultFile.write(im_name + '\n')
    resultFile.write(str(len(dets)) + '\n')
    if len(dets) != 0:
        for det in dets:
            bbox = det[:4]
            score = det[-1]
            resultFile.write('{:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                             format(score, bbox[0], bbox[1], bbox[2], bbox[3]))
    resultFile.close()

def load_detections(filePath):
    with open(filePath, 'r') as f:
        lines = f.readlines()
    index = 0
    detImgInfos = {}
    while index < len(lines):
        im_name = lines[index][:-1]
        roiNum = int(lines[index+1][:-1])
        if roiNum == 0:
            detImgInfos[im_name] = []
        else:
            im_info = np.array([lines[j][:-1].split(' ') for j in range(index + 2, index + 2 + roiNum)], dtype=float)
            detImgInfos[im_name] = np.hstack([im_info[:, 1:], im_info[:, :1]])  # (box, score)

        index = index + 2 + roiNum

    return detImgInfos


def load_gt_detections(imdb):
    gtImgInfos = {}
    for roidb in imdb.roidb:
        im_name = os.path.splitext(os.path.basename(roidb['image']))[0]
        boxes = roidb['boxes']
        gtImgInfos[im_name] = boxes
    return gtImgInfos


def cal_ap_mp_voc(annoImgInfos, detImgInfos, overthresh=0.5, recordIMFP=0):
    # prepare make annotated image
    oriInfos = {}
    oriDet = {}
    oriRoiSum = 0
    for im_name in annoImgInfos.keys():
        oriImgBoxs = annoImgInfos[im_name]
        oriDet[im_name] = [False] * len(oriImgBoxs)
        oriInfos[im_name] = oriImgBoxs
        oriRoiSum += len(oriImgBoxs)

    key_ids = np.array(detImgInfos.keys())
    image_roiNum = np.array([len(detImgInfos[id]) for id in key_ids])
    valid_indexs = np.where(image_roiNum != 0)[0]
    # when nothing to detect
    if len(valid_indexs) == 0:
        ret = [0] * 8
        ret.append('oriAnnNum:0 detNum:0 det_TP:0 det_FP:0 det_AP:0.0')
        print ret[-1]
        return ret
    key_ids = key_ids[valid_indexs]
    image_roiNum = image_roiNum[valid_indexs]
    image_ids = np.repeat(key_ids, image_roiNum)
    confidence = np.hstack([detImgInfos[id][:, -1] for id in key_ids])
    BB = np.vstack([detImgInfos[id][:, :-1] for id in key_ids])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = (-1) * np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = image_ids[sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = oriInfos[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R.astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > overthresh:
            # if not R['difficult'][jmax]:
            if not oriDet[image_ids[d]][jmax]:
                tp[d] = 1.
                oriDet[image_ids[d]][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    # plot FP or TP
    # if recordIMFP:
    #     plot_IM_FPTP(fp, tp, sorted_scores)
    # compute precision recall
    fp_sum = sum(fp)
    tp_sum = sum(tp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(oriRoiSum)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, False)  # use_07_metric

    print('oriAnnNum:%d detNum:%d det_TP:%d det_FP:%d det_AP:%f' % (oriRoiSum, nd, tp_sum, fp_sum, ap))
    str = 'oriAnnNum:%d detNum:%d det_TP:%d det_FP:%d det_AP:%f' % (oriRoiSum, nd, tp_sum, fp_sum, ap)
    return [ap, fp_sum, tp_sum, fp, tp, rec, prec, oriRoiSum, str]


def test_net(net, imdb, output_dir, vis=0):
    """Test RON network on an image database."""
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    # output_dir = get_output_dir(imdb, net)  # testDB + '_test'
    output_dir = get_dir(os.path.join(output_dir, imdb.name))
    print 'Output will be saved to `{:s}`'.format(output_dir)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    for i in xrange(0, num_images, cfg.TEST.BATCH_SIZE):
        _t['misc'].tic()
        ims = []
        for im_i in xrange(cfg.TEST.BATCH_SIZE):
            im = cv2.imread(imdb.image_path_at(i + im_i))
            ims.append(im)
        _t['im_detect'].tic()
        batch_scores, batch_boxes = im_detect(net, ims)
        _t['im_detect'].toc()

        for im_i in xrange(cfg.TEST.BATCH_SIZE):
            im = ims[im_i]
            scores = batch_scores[im_i]
            boxes = batch_boxes[im_i]

            # filter boxes according to prob scores
            keeps = np.where(scores[:, 0] > cfg.TEST.PROB)[0]
            scores = scores[keeps, :]
            boxes = boxes[keeps, :]

            im_shape = im.shape[0:2]
            if cfg.TEST.ADAPT_SCALE:
                boxes[:, :] = boxes[:, :] / cfg.TEST.IMAGE_SCALE
            else:
                # change boxes according to input size and the original image size
                im_scales = float(cfg.TEST.SCALES[0]) / np.array(im_shape)
                boxes[:, 0::2] = boxes[:, 0::2] / im_scales[1]
                boxes[:, 1::2] = boxes[:, 1::2] / im_scales[0]

            # filter boxes with small sizes
            boxes = clip_boxes(boxes, im_shape)
            keep = filter_boxes(boxes, cfg.TEST.RON_MIN_SIZE)
            scores = scores[keep, :]
            boxes = boxes[keep, :]

            scores = np.tile(scores[:, 0], (imdb.num_classes, 1)).transpose() * scores

            for j in xrange(1, imdb.num_classes):
                inds = np.where(scores[:, j] > cfg.TEST.DET_MIN_PROB)[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, :]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)

                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep, :]
                if len(keep) > cfg.TEST.BOXES_PER_CLASS:
                    cls_dets = cls_dets[:cfg.TEST.BOXES_PER_CLASS, :]
                all_boxes[j][i + im_i] = cls_dets

                if vis:
                    vis_detections(im, imdb.classes[j], cls_dets)
            _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
            .format(i + 1, num_images, _t['im_detect'].average_time,
                    _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)


def _delete_file(file):
    if os.path.exists(file):
        os.remove(file)

def mk_dirs(saveDir, subDirs):
    objectDirs = []
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    else:
        shutil.rmtree(saveDir)
        os.makedirs(saveDir)
    for subDir in subDirs:
        objectDir = os.path.join(saveDir, subDir)
        os.makedirs(objectDir)
        objectDirs.append(objectDir)
    return objectDirs


def test_net_det(net, imdb, output_dir, recompute=1, vis=0, record=1, overthresh=0.5):
    """Test network on wider database."""
    output_dir = get_dir(os.path.join(output_dir, imdb.name))
    print 'Output will be saved to `{:s}`'.format(output_dir)
    resultFilePath = os.path.join(output_dir, 'Result_%s.txt' % net.name)

    if not os.path.exists(resultFilePath) or recompute:
        # delete previous file
        _delete_file(resultFilePath)
        # timers
        num_images = len(imdb.image_index)
        all_boxes = [[] for _ in xrange(num_images)]
        _t = {'im_detect': Timer(), 'misc': Timer()}
        for i in xrange(0, num_images, cfg.TEST.BATCH_SIZE):
            _t['misc'].tic()
            ims = []
            for im_i in xrange(cfg.TEST.BATCH_SIZE):
                im = cv2.imread(imdb.image_path_at(i + im_i))
                ims.append(im)
            _t['im_detect'].tic()
            batch_scores, batch_boxes = im_detect(net, ims)
            _t['im_detect'].toc()

            for im_i in xrange(cfg.TEST.BATCH_SIZE):
                im = ims[im_i]
                scores = batch_scores[im_i]
                boxes = batch_boxes[im_i]

                # filter boxes according to prob scores
                keeps = np.where(scores[:, 0] > cfg.TEST.PROB)[0]
                scores = scores[keeps, :]
                boxes = boxes[keeps, :]

                # change boxes according to input size and the original image size
                # im_shape = im.shape[0:2]
                # im_scales = float(cfg.TEST.SCALES[0]) / np.array(im_shape)
                #
                # boxes[:, 0::2] = boxes[:, 0::2] / im_scales[1]
                # boxes[:, 1::2] = boxes[:, 1::2] / im_scales[0]

                im_shape = im.shape[0:2]
                if cfg.TEST.ADAPT_SCALE:
                    boxes[:, :] = boxes[:, :] / cfg.TEST.IMAGE_SCALE
                else:
                    # change boxes according to input size and the original image size
                    im_scales = float(cfg.TEST.SCALES[0]) / np.array(im_shape)
                    boxes[:, 0::2] = boxes[:, 0::2] / im_scales[1]
                    boxes[:, 1::2] = boxes[:, 1::2] / im_scales[0]

                # filter boxes with small sizes
                boxes = clip_boxes(boxes, im_shape)
                keep = filter_boxes(boxes, cfg.TEST.RON_MIN_SIZE)
                scores = scores[keep, :]
                boxes = boxes[keep, :]

                # scores = np.tile(scores[:, 0], (imdb.num_classes, 1)).transpose() * scores

                cls_dets = np.hstack((boxes, scores)).astype(np.float32, copy=False)

                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep, :]
                # if len(keep) > cfg.TEST.BOXES_PER_CLASS:
                #     cls_dets = cls_dets[:cfg.TEST.BOXES_PER_CLASS, :]
                all_boxes[i + im_i] = cls_dets

                # if vis:
                #     vis_detections(im, imdb.classes[j], cls_dets)

                if record:
                    imName = os.path.splitext(os.path.basename(imdb.image_path_at(i + im_i)))[0]
                    record_detections(resultFilePath, imName, imdb.classes[1], cls_dets, thresh=cfg.TEST.DET_MIN_PROB)
            _t['misc'].toc()

            print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                .format(i + 1, num_images, _t['im_detect'].average_time,
                        _t['misc'].average_time)
    # exit(1)
    # load detect file
    detImgInfos = load_detections(resultFilePath)
    print '{} detected results loaded from {}'.format(imdb.name, resultFilePath)

    # load gt file
    annoImgInfos = load_gt_detections(imdb)

    # vis result
    if vis:
        output_dir = get_dir(os.path.join(output_dir, 'DetImg_{:s}'.format(net.name)))
        print 'Image Output will be saved to `{:s}`'.format(output_dir)
        visual_result_det(annoImgInfos, detImgInfos, imdb, output_dir, visualThreshold=cfg.TEST.PROB)

    metric = cal_ap_mp_voc(annoImgInfos, detImgInfos, overthresh=overthresh)

    return metric


# used to compute wider benchmark
def test_net_eva(net, widerDir, output_dir, recompute=1):
    """Test network on wider database."""
    print 'Output will be saved to `{:s}`'.format(output_dir)

    if not os.path.exists(output_dir) or recompute:
        # delete previous file
        # shutil.rmtree(output_dir)
        # timers
        event_list = os.listdir(widerDir)
        file_list = [os.listdir(os.path.join(widerDir, event)) for event in event_list]
        objectDirs = mk_dirs(output_dir, event_list)
        num_images = sum([len(file) for file in file_list])
        record_num = 0
        _t = {'im_detect': Timer(), 'misc': Timer()}
        for objectDir, event, files in zip(objectDirs, event_list, file_list):
            for num, file in enumerate(files):
                _t['misc'].tic()
                ims = []
                file = file[:-4]
                imPath = os.path.join(widerDir, event, file + '.jpg')
                detectFilePath = os.path.join(objectDir, file + '.txt')
                resultList = open(detectFilePath, 'w')
                im = cv2.imread(imPath)
                ims.append(im)

                _t['im_detect'].tic()
                batch_scores, batch_boxes = im_detect(net, ims)
                _t['im_detect'].toc()

                for im_i in xrange(cfg.TEST.BATCH_SIZE):
                    im = ims[im_i]
                    scores = batch_scores[im_i]
                    boxes = batch_boxes[im_i]

                    # filter boxes according to prob scores
                    keeps = np.where(scores[:, 0] > cfg.TEST.PROB)[0]
                    scores = scores[keeps, :]
                    boxes = boxes[keeps, :]

                    # change boxes according to input size and the original image size
                    # im_shape = im.shape[0:2]
                    # im_scales = float(cfg.TEST.SCALES[0]) / np.array(im_shape)
                    #
                    # boxes[:, 0::2] = boxes[:, 0::2] / im_scales[1]
                    # boxes[:, 1::2] = boxes[:, 1::2] / im_scales[0]

                    im_shape = im.shape[0:2]
                    if cfg.TEST.ADAPT_SCALE:
                        boxes[:, :] = boxes[:, :] / cfg.TEST.IMAGE_SCALE
                    else:
                        # change boxes according to input size and the original image size
                        im_scales = float(cfg.TEST.SCALES[0]) / np.array(im_shape)
                        boxes[:, 0::2] = boxes[:, 0::2] / im_scales[1]
                        boxes[:, 1::2] = boxes[:, 1::2] / im_scales[0]

                    # filter boxes with small sizes
                    boxes = clip_boxes(boxes, im_shape)
                    keep = filter_boxes(boxes, cfg.TEST.RON_MIN_SIZE)
                    scores = scores[keep, :]
                    boxes = boxes[keep, :]

                    cls_dets = np.hstack((boxes, scores)).astype(np.float32, copy=False)

                    keep = nms(cls_dets, cfg.TEST.NMS)
                    dets = cls_dets[keep, :]
                    # record result
                    title_name = event + '/' + file + '.jpg'
                    resultList.write(title_name + '\n')
                    resultList.write(str(dets.shape[0]) + '\n')
                    if dets.shape[0] == 0:
                        continue
                    for i in range(0, dets.shape[0]):
                        bbox = dets[i, :4]
                        score = dets[i, -1]
                        resultList.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2] - bbox[0]) + ' ' +
                                         str(bbox[3] - bbox[1]) + ' ' + str(score) + '\n')
                    resultList.close()
                _t['misc'].toc()
                # print('event:%s num:%d' % (event, num + 1))

                print 'event: {:s} num: {:d} im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                    .format(event, num + 1, record_num + num + 1, num_images, _t['im_detect'].average_time,
                            _t['misc'].average_time)
            record_num = record_num + num + 1

    return 'done'


def test_net_gen(net, imdb, output_dir, record=1, overthresh=0.5, vis=0):
    if imdb.name.find('voc') > -1:
        test_net(net, imdb, output_dir)
    elif imdb.name.find('wider') > -1:
        metricStr = test_net_det(net, imdb, output_dir, record, overthresh=overthresh, vis=vis)

    return metricStr[-1]


def visual_result_det(oriImgInfos, detImgInfos, imdb, output, visualThreshold=0.5):
    for roidb in imdb.roidb:
        # visual annotation
        im_name = os.path.splitext(os.path.basename(roidb['image']))[0]
        annoImgBoxs = np.array(oriImgInfos[im_name], dtype=np.float32)
        path = os.path.join(cfg.ROOT_DIR, roidb['image'].replace('../', ''))
        im = cv2.imread(path)

        for i in range(len(annoImgBoxs)):
            x1 = annoImgBoxs[i][0]
            y1 = annoImgBoxs[i][1]
            x2 = annoImgBoxs[i][2]
            y2 = annoImgBoxs[i][3]
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # visual detection
        detImgBoxs = np.array(detImgInfos[im_name], dtype=np.float32)
        if len(detImgBoxs) != 0:
            for i in range(len(detImgBoxs)):
                x1 = detImgBoxs[i][0]
                y1 = detImgBoxs[i][1]
                x2 = detImgBoxs[i][2]
                y2 = detImgBoxs[i][3]
                score = detImgBoxs[i][-1]
                scale = 0.8  # (10/(bbox[2]-bbox[0]))
                if score >= visualThreshold:
                    cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    # cv2.putText(im, 'face', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)
                    cv2.putText(im, str(score), (x1, int(y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2)

        imfile = output + '/re_' + '_'.join(im_name.split('/')) + '.jpg'
        cv2.imwrite(imfile, im)
    pass
