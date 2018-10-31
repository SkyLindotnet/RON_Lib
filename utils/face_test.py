#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import _init_paths
from fast_rcnn.config import cfg
# from fast_rcnn.test import im_detect, im_detect_rpn
# import rpn.generate
from fast_rcnn.nms_wrapper import nms #, soft_nms
import numpy as np
import caffe, os, cv2
import shutil
from mylab.draw import *
from datasets.factory import get_imdb
from fast_rcnn.test import test_net_gen, test_net_eva

CLASSES = ('__background__',
           'face')
widerValImgList = cfg.ROOT_DIR + '/data/DB/object/voc_wider/ImageSets/Main/val.txt'
widerValAnnoPath = cfg.ROOT_DIR + '/data/DB/object/voc_wider/Annotations/{:s}.xml'
matFile = cfg.ROOT_DIR + '/data/DB/face/wider/wider_face_val-v7.mat'

class customError(StandardError):
    pass

# region auxiliary methods

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

def mk_dir(saveDir, cover=1):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    elif cover:
        shutil.rmtree(saveDir)
        os.makedirs(saveDir)

def soft_bbox_vote(det):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.45)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= 0.01)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((soft_det_accu, det_accu_sum))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]
    return dets

# endregion


# region generate benchmark

def load_net(methodName, prototxt, modelPath):
    # load initial parameter
    # cfg.TEST.HAS_RPN = True
    # initialize net
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)
    # load net
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    net.name = os.path.splitext(os.path.basename(str(methodName)))[0]
    return net


def det_model_wider(net, widerDir, saveDir):

    event_list = os.listdir(widerDir)
    file_list = [os.listdir(os.path.join(widerDir, event)) for event in event_list]
    objectDirs = mk_dirs(saveDir, event_list)
    for objectDir, event, files in zip(objectDirs, event_list, file_list):
        for num, file in enumerate(files):
            file = file[:-4]
            imPath = os.path.join(widerDir, event, file + '.jpg')
            detectFilePath = os.path.join(objectDir, file + '.txt')
            resultList = open(detectFilePath, 'w')
            im = cv2.imread(imPath)

            all_dets = []
            for scale in cfg.TEST.SCALES:
                cfg.TEST.SCALES = [base_scales + scale]  # 600 800 1000 1200 1400
                cfg.TEST.MAX_SIZE = base_max_size + scale
                if modelType == 'rpn':
                    boxes, scores = rpn.generate.im_proposals(net, im.copy())
                    cls_boxes = boxes[:, 0:4]
                    cls_scores = scores[:, 0]
                elif modelType == 'normal':
                    scores, boxes = im_detect(net, im.copy())
                    cls_boxes = boxes[:, 4:8]
                    cls_scores = scores[:, 1]
                elif modelType == 'normal_rpn':
                    scores, boxes = im_detect_rpn(net, im.copy())
                    cls_boxes = boxes[:, 0:4]
                    cls_scores = scores[:, 0]

                dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                keep = soft_nms(dets, sigma=0.5, Nt=NMS_THRESH, threshold=0.01, method=1)
                dets = dets[keep, :]
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

                if all_dets == []:
                    all_dets = dets[inds, :].copy()
                else:
                    all_dets = np.vstack((all_dets, dets[inds, :]))

            dets = soft_bbox_vote(all_dets)

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
            print('event:%s num:%d' % (event, num + 1))


def generate_benchmark_result(methodName, prototxt, modelPath, testDataSet, resultSaveDir, recompute=1):
    # load net
    net = load_net(methodName, prototxt, modelPath)
    if testDataSet.find('wider') != -1:
        # detect and record in wider face format
        dataset = 'val' if testDataSet.find('val') != -1 else 'test'
        widerDir = cfg.ROOT_DIR + '/data/DB/face/wider/%s/' % dataset  # val test
        widerDetectSaveDir = os.path.join(resultSaveDir, net.name)
        test_net_eva(net, widerDir, widerDetectSaveDir, recompute)

# endregion


# region test model

def det_model_voc_pyramid(prototxt, modelPath, modelType, im_names, vocDetectFilePath,
                          vocValImgDir, CONF_THRESH=0.8, NMS_THRESH=0.3, scales=[0],
                          testingDB='wider',includeRPN=0):
    # load net
    net = caffe.Net(str(prototxt), str(modelPath), caffe.TEST)
    resultList = open(vocDetectFilePath, 'w')
    if includeRPN:
        resultRPNList = open(vocDetectFilePath.replace('.txt', '_RPN.txt'), 'w')
    base_scales = 600
    base_max_size = 1000
    for im_name in im_names:
        # Load the demo image
        image_name = im_name[:-1]

        im_path = os.path.join(vocValImgDir, image_name + '.jpg')
        im = cv2.imread(im_path)

        # Detect all object classes and regress object bounds
        boxesList = []
        scoresList = []
        boxesRPNList = []
        scoresRPNList = []
        for scale in scales:
            cfg.TEST.SCALES = [base_scales + scale]  # 600 800 1000 1200 1400
            cfg.TEST.MAX_SIZE = base_max_size + scale
            if modelType == 'rpn':
                boxes, scores = rpn.generate.im_proposals(net, im)
            # elif modelType == 'cascade':
            #     scores, boxes = im_detect_by_rois(net, im.copy(), rois_layer='fc_rois')
            else:
                if includeRPN:
                    scores, boxes, scoresRPN, boxesRPN = im_detect(net, im.copy(), includeRPN=True)
                else:
                    scores, boxes = im_detect(net, im.copy())

                if cfg.PYRAMID_DETECT_DEBUG:
                    startIndex = 0 + cfg.PYRAMID_DETECT_NUM * cfg.PYRAMID_DETECT_INDEX
                    endIndex = startIndex + cfg.PYRAMID_DETECT_NUM - 1
                    scores = scores[startIndex:endIndex]
                    boxes = boxes[startIndex:endIndex]

            boxesList.append(boxes.copy())
            scoresList.append(scores.copy())
            if includeRPN:
                boxesRPNList.append(boxesRPN.copy())
                scoresRPNList.append(scoresRPN.copy())

        for cls_ind, cls in enumerate(CLASSES[1:]):
            detsList = []
            if modelType == 'normal':
                cls_ind += 1  # because we skipped background
            for boxes, scores in zip(boxesList, scoresList):
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                if detsList == []:
                    detsList = dets[inds, :].copy()
                else:
                    detsList = np.vstack((detsList, dets[inds, :]))

            # dets with different scale do nms
            keep = nms(detsList, NMS_THRESH)
            dets = detsList[keep, :]

            # record result
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

            if len(inds) == 0:
                continue
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                if testingDB == 'FDDB':
                    resultList.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2] - bbox[0]) + ' ' +
                                     str(bbox[3] - bbox[1]) + ' ' + str(score) + '\n')
                elif testingDB == 'wider':
                    resultList.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                     format(image_name, score,
                                            bbox[0] + 1, bbox[1] + 1,
                                            bbox[2] + 1, bbox[3] + 1))
                else:
                    raise customError('testingDB is invalid')

        if includeRPN:
            for cls_ind, cls in enumerate(CLASSES[1:]):
                detsList = []
                for boxes, scores in zip(boxesRPNList, scoresRPNList):
                    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                    cls_scores = scores[:, cls_ind]
                    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                    keep = nms(dets, NMS_THRESH)
                    dets = dets[keep, :]
                    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                    if detsList == []:
                        detsList = dets[inds, :].copy()
                    else:
                        detsList = np.vstack((detsList, dets[inds, :]))

                # dets with different scale do nms
                keep = nms(detsList, NMS_THRESH)
                dets = detsList[keep, :]

                # record result
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

                if len(inds) == 0:
                    continue
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    if testingDB == 'FDDB':
                        resultRPNList.write(str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2] - bbox[0]) + ' ' +
                                         str(bbox[3] - bbox[1]) + ' ' + str(score) + '\n')
                    elif testingDB == 'wider':
                        resultRPNList.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                         format(image_name, score,
                                                bbox[0] + 1, bbox[1] + 1,
                                                bbox[2] + 1, bbox[3] + 1))
                    else:
                        raise customError('testingDB is invalid')

    resultList.close()
    if includeRPN:
        resultRPNList.close()


def det_test_wider(net, modelType, saveDir, CONF_THRESH=0.8, NMS_THRESH=0.3, scales=[0]):
    pass


def generate_test_result(methodName, prototxt, modelPath, testDataSet, resultSaveDir, recompute=1,
                         overthresh=0.5, vis=0):
    # load net
    net = load_net(methodName, prototxt, modelPath)
    imdb = get_imdb(testDataSet)
    metric_str = test_net_gen(net, imdb, resultSaveDir, recompute, overthresh=overthresh, vis=vis)
    print metric_str
    # record result
    path = os.path.join(resultSaveDir, imdb.name, 'Metric_%s.txt' % net.name)
    MetricFile = open(path, 'w')
    MetricFile.write('%s\n' % metric_str)
    MetricFile.close()

# endregion

if __name__ == '__main__':

    # region parameter initialization
    cfg.TEST.SCALES = (640,)  # 640 scales = [0, 200, 400, 600, 800, 1000]
    cfg.TEST.PROB = 0.8  # 0.03
    cfg.TEST.NMS = 0.3  # 0.4
    cfg.TEST.RON_MIN_SIZE = 0.1  # 10

    # cfg.TEST.ADAPT_SCALE = 1
    # cfg.TEST.MIN_MAX_SCALES = [[640, 1000]]

    cfg.GPU_ID = 2
    # method = 'wider/VGG16-ms-fpns-v1-2-reduce_3(4-5-6d_2)(norm)_adapt_640x1000_st(ssh1)_b3_t_iter_160000'
    method = 'wider/VGG16-ms-rpns-v2-reduce_1(5)_640x640_st(ssh1)_gpu1_iter_160000'
    prototxt = 'wider/VGG16-MS-RPN/test_ms-rpns-v2-reduce_1(5)'
    # method = 'wider/VGG16_faster_rcnn_end2end_with_fuse_multianchor-ms-rpns_v2-4-1_voc_fp3_v3-10_m0.5_2_1_ms-rpns-v2-reduce_2(norm)-rois-norm-10-8-5_o_iter_200000'
    # prototxt = 'wider/VGG16-MS-RPN/test_fuse_multianchor_v2-4-1-ms-rpns-v2-reduce_2(norm)-roi-norm-10-8-5'
    # [d3-k(3-5)]
    testDataSet = 'wider_val'
    testType = 'benchmark'  # normal benchmark
    suffix = 'VAL'  # OT(0.7)_
    # endregion

    # region filePath auto match
    scale_str = '_'.join(np.array(cfg.TEST.MIN_MAX_SCALES[0], dtype=np.str)) if cfg.TEST.ADAPT_SCALE else '_'.join(np.array(cfg.TEST.SCALES, dtype=np.str))
    suffix = '_c(%s)_n(%s)_s(%s)_%s' % (str(cfg.TEST.PROB).replace('.', ''),
                                        str(cfg.TEST.NMS).replace('.', ''),
                                        scale_str, suffix)
    methodName = method + suffix
    modelPath = os.path.join(cfg.ROOT_DIR, 'output/%s' % method + '.caffemodel')
    prototxt = os.path.join(cfg.ROOT_DIR, 'models/%s' % prototxt + '.prototxt')
    # endregion

    if testType == 'normal':
        resultSaveDir = os.path.join(cfg.ROOT_DIR, 'output/result/%s' % testType)
        generate_test_result(methodName, prototxt, modelPath, testDataSet,
                             resultSaveDir, recompute=1, overthresh=0.5,
                             vis=0)
    else:
        resultSaveDir = os.path.join(cfg.ROOT_DIR, 'output/result/%s/%s' % (testType, testDataSet))
        generate_benchmark_result(methodName, prototxt, modelPath, testDataSet,
                             resultSaveDir, recompute=1)
    print 'done'

