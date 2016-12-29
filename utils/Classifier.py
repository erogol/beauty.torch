"""
A interface for Torch models
"""
import lutorpy as lua
import numpy as np
import os
import sys
import pandas as pd
import cPickle
import time

lua.require('nn')
lua.require('cunn')
lua.require('cudnn')
lua.eval("torch.setdefaulttensortype('torch.FloatTensor')")

import logging
from skimage import io, transform
from matplotlib import pylab as plt

def thumbnail(img, size=150):
    """
        Resize given image by the shortest edge
    """
    from math import floor
    width, height = img.shape[1], img.shape[0]

    if width == height:
        img = transform.resize(img, [size, size])

    elif height > width:
        ratio = float(height)/float(width)
        newheight = ratio * size
        img = transform.resize(img, [int(floor(newheight)), size ])

    elif width > height:
        ratio = float(width) / float(height)
        newwidth = ratio * size
        img = transform.resize(img, [size, int(floor(newwidth))])

    img = (img*255).astype('uint8')
    return img

def crop_center(img):
    """
        Crop image by center with the size of shortest edge
    """
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    return crop_img



class TorchModel(object):
    '''
    Parent class for all torch Models
    '''

    def __init__(self, gpu_mode=0, model_path=None, model_file=None, image_size=224, mean=None, std=None, classification=True):
        """
        Parameters
        ----------
        gpu_mode : bool
            If True model runs on GPU
        crop_center : bool, optional
            if True, model crops the image center by resizing the image regarding
            shortest side.
        is_retrieval : bool, optional
            if True, model constructs feature extractor.
        """
        print model_file

        ROOT_PATH = model_path
        self.gpu_mode = gpu_mode
        self.classification = classification

        # Load the pre-trained model
        if model_file ==  None:
            model_path = os.path.join(ROOT_PATH,"model_cpu.t7")
        else:
            model_path = os.path.join(ROOT_PATH, model_file)
        self.model = torch.load(model_path)

        if self.classification:
            self.model._add(nn.SoftMax())

        if self.gpu_mode:
            self.model._cuda()

        self.model._evaluate()

        # Load mean file
        if mean is None:
            self.mean = np.array([ 0.485,  0.456,  0.406])
        else:
            self.mean = mean

        if std is None:
            self.std  = np.array([ 0.229,  0.224,  0.225])
        else:
            self.std = std

        # Load synset (text label)
        try:
            self.synset = [l.split(',')[0].strip() for l in open(ROOT_PATH+'synset.txt').readlines()]
        except:
            pass
        self.image_size = image_size

    def preprocess_image(self, img):
        if type(img) is str or type(img) is unicode:
            img = io.imread(img)
        else:
            img = img

        # resize image by shortest edge
        img_resized = thumbnail(img, self.image_size)/ float(255)

        # color normalization
        img_norm = img_resized - self.mean
        img_norm /= self.std

        # center cropping
        img_crop = crop_center(img_norm)

        # format img dimensions
        img_crop = img_crop.transpose([2,0,1])[None,:]

        assert img_crop.ndim == 4

        # pass data to torch and convert douple to float
        x = torch.fromNumpyArray(img_crop)
        x = x._float()
        return x

    def classify_image(self, img, N=2):
        start = time.time()
        img = self.preprocess_image(img)
        # Get prediction probability of 1000 classes from model
        if self.gpu_mode:
            prob = self.model._forward(img._cuda())
        else:
            prob = self.model._forward(img)
        prob = prob.asNumpyArray()[0]
        end   = time.time()
        # Argsort, get prediction index from largest prob to lowest
        pred = np.argsort(prob)[::-1]
        # Get topN label
        topN = [self.synset[pred[i]] for i in range(N)]
        topN_probs = [float(p) for p in prob[pred[0:N]]]
        topN = zip(topN, pred[0:N])
        topN = [ topN[c] + (topN_prob,) for c,topN_prob in enumerate(topN_probs)]
        #print "WQETQWETQWERQWERQWERQWER", prob[pred[0:5]]
        return '%.3f' % (end - start), topN


class HotOrNotTorchRegression(TorchModel):
    """
    Interface for pretrained mxnet inception model on 1000 concepts used by ImageNet
    challenge. It is able to extract features and classify the given image
    """
    def __init__(self, gpu_mode, model_path, model_file, image_size=224, mean=None, std=None):
        super(HotOrNotTorchRegression, self).__init__(gpu_mode, model_path=model_path, model_file=model_file, image_size=image_size, mean=mean, std=std, classification=False)

    def classify_image(self, img, N=10):
        start = time.time()
        img = self.preprocess_image(img)
        # Get prediction probability of 1000 classes from model
        if self.gpu_mode:
            score = self.model._forward(img._cuda())
        else:
            score = self.model._forward(img)
        score = score.asNumpyArray()[0][0]
        end   = time.time()
        return '%.3f' % (end - start), score
