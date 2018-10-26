# --------------------------------------------------------
# RON
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and modified by Tao Kong
# --------------------------------------------------------

"""Train RON network."""

import caffe
from fast_rcnn.config import cfg
from utils.timer import Timer
import numpy as np
import os
from caffe.proto import caffe_pb2
import google.protobuf as pb2
from tempfile import NamedTemporaryFile
from datasets.factory import get_imdb
from test import test_net_gen
import google.protobuf.text_format


class SolverWrapper(object):
    """
    A simple wrapper around Caffe's solver.
    """

    def __init__(self, solver_prototxt, roidb, output_dir, version,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.version = version
        self.hasRecorded = 0
        # custom
        # judge solverstate
        if cfg.TRAIN.WITH_SOLVERSTATE:
            self.solver_param = caffe_pb2.SolverParameter()
            with open(solver_prototxt, 'rt') as f:
                pb2.text_format.Merge(f.read(), self.solver_param)
            infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
            baseFileName = (self.solver_param.snapshot_prefix + infix + '_{:s}'.format(version))
            self.solver_param.snapshot_prefix = os.path.join(self.output_dir, baseFileName)
            with NamedTemporaryFile('w', delete=False) as f:
                f.write(pb2.text_format.MessageToString(self.solver_param))
            self.solver = caffe.SGDSolver(f.name)  # rename solverstate
        else:
            self.solver = caffe.SGDSolver(solver_prototxt)  # all layers from python module will be setup
        # load pretrained_model or solverstate
        if pretrained_model is not None:
            if pretrained_model.endswith('.caffemodel'):
                print ('Loading pretrained model '
                       'weights from {:s}').format(pretrained_model)
                self.solver.net.copy_from(pretrained_model)
            elif pretrained_model.endswith('.solverstate'):
                print ('Loading solverstate'
                       ' from {:s}').format(pretrained_model)
                self.solver.restore(pretrained_model)
                self.hasRecorded = 1
        # if model is not None:
        #     print ('Loading pretrained model '
        #            'weights from {:s}').format(model)
        #     self.solver.net.copy_from(model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        # Initialize record parameter
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        baseFilename = (self.solver_param.snapshot_prefix + infix +
                        '_{:s}'.format(version) + '.txt')
        self.recordLossFile = os.path.join(self.output_dir, 'Loss_' + baseFilename)
        self.recordMetricFile = os.path.join(self.output_dir, 'Metric_' + baseFilename)

        self.solver.net.layers[0].set_roidb(roidb)

    def snapshot(self):
        """
        Take a snapshot of the network.
        """
        net = self.solver.net

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix + '_{:s}'.format(self.version) +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        if cfg.TRAIN.WITH_SOLVERSTATE:
            self.solver.snapshot()
        else:
            net.save(str(filename))
            print 'Wrote snapshot to: {:s}'.format(filename)

        print 'have snapshoted'
        return filename

    def snapshot_temp(self):
        """
        Take a snapshot of the network.
        """
        net = self.solver.net
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = '_' + str(cfg.TRAIN.SCALES[0])
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        return filename

    def train_model_temp(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()

    def train_model(self, max_iters):
        """Network training loop."""
        timer = Timer()

        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)
            # record loss and metric
            if self.solver.iter % cfg.TRAIN.LOSS_ITERS == 0:
                self.recordLoss()
            if self.solver.iter % cfg.TRAIN.METRIC_ITERS == 0:
                self.recordMetric()
            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                self.last_snapshot_iter = self.solver.iter
                self.snapshot()

        if self.last_snapshot_iter != self.solver.iter:
            self.snapshot()

    def recordLoss(self):
        # self.solver.iter
        if self.hasRecorded:
            LossFile = open(self.recordLossFile, 'a')
        else:
            LossFile = open(self.recordLossFile, 'w+')
            self.hasRecorded = 1

        net = self.solver.net
        lossStr = ''
        lossSum = 0
        iterStr = 'Iter_{:d}'.format(self.solver.iter)
        lossNameList = filter(lambda str: str.find('loss') != -1, net.blobs.keys())
        for lossName in lossNameList:
            lossStr += lossName + ' %.5f ' % net.blobs[lossName].data
            lossSum += net.blobs[lossName].data

        LossFile.write(iterStr + ' ' + lossStr + 'allLoss %.8f\n' % lossSum)
        LossFile.close()

    def recordMetric(self):
        MetricFile = open(self.recordMetricFile, 'a')

        net = self.solver.net
        # tempModel used to metric
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        methodName = self.solver_param.snapshot_prefix + infix + '_{:s}'.format(self.version) \
                     + '_iter_{:d}'.format(self.solver.iter)
        tempModelPath = os.path.join(self.output_dir, 'Temp_' + methodName + '.caffemodel')
        net.save(str(tempModelPath))
        # start metric
        prototxt = cfg.TRAIN.TestPrototxt
        caffe.set_mode_gpu()
        caffe.set_device(cfg.GPU_ID)
        net = caffe.Net(prototxt, str(tempModelPath), caffe.TEST)
        net.name = os.path.splitext(os.path.basename(str(tempModelPath)))[0]
        imdb = get_imdb(cfg.TRAIN.TestDataSet)
        # ap = generate_result_wider_val(tempModelPath, prototxt, methodName, methodType)[0]
        # metric_str = generate_result_voc_val(tempModelPath, prototxt, methodName, methodType)
        metric_str = test_net_gen(net, imdb, self.output_dir)
        print metric_str
        os.remove(tempModelPath)
        iterStr = '{:s}_Iter_{:d}'.format(imdb.name, self.solver.iter)
        MetricFile.write(iterStr + ' ' + '%s\n' % metric_str)
        MetricFile.close()


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'
    print 'done'
    return imdb.roidb


def train_net(solver_prototxt, roidb, output_dir, version,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, roidb, output_dir, version,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'
