# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:30:20 2020

@author: Pranjal Vithlani
"""

import torch
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import json
import pdb


class ImageEncoder(nn.Module):
    def __init__(self, modeltype):
        """Load the pretrained model and replace top fc layer."""
        super(ImageEncoder, self).__init__()
        if modeltype == 'resnet152':
            self.ImageEnc = models.resnet152(pretrained=True)
        elif modeltype == 'resnet101':
            self.ImageEnc = models.resnet101(pretrained=True)
        elif modeltype == 'resnet50':
            self.ImageEnc = models.resnet50(pretrained=True)
        elif modeltype == 'resnet18':
            self.ImageEnc = models.resnet18(pretrained=True)
        else:
            raise ValueError('{} not supported'.format(modeltype))
        self.layer = self.ImageEnc._modules.get('avgpool')
        self.ImageEnc = nn.Sequential(*list(
                        self.ImageEnc.children())[:-1])
        self.ImageEnc.eval()

    def forward(self, images):
        """Extract the image feature vectors."""
        my_embedding = self.ImageEnc(images).squeeze()

        return my_embedding


class ExtractImageFeat():
    def __init__(self, image_feat_type):
        super(ExtractImageFeat, self).__init__()
        self.cnnmodel = ImageEncoder(image_feat_type).cuda()
        self.image_loader = transforms.Compose(
                [transforms.Resize(224),
                 transforms.CenterCrop(224), transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])

    def get_feat_per_img(self, image_file):
        isvalid = False
        try:
            image = Image.open(image_file).convert('RGB')
            image_vec = self.image_loader(image).float()
            image_vec = image_vec.cuda()
            featvec = self.cnnmodel(
                    image_vec.unsqueeze(0)).detach().cpu().numpy()
            isvalid = True
        except:
            featvec = None
            isvalid = False
        return featvec, isvalid

    def get_all_feats(self):

        if not os.path.isdir(trailer_feat_savedir+movie_name):
            os.makedirs(trailer_feat_savedir+movie_name)
        
        totframes = 0
        processed_frames = 0
        error_frames = 0
        isvalid = False
        datafile = os.listdir(input_dir+movie_name)
        totsamples = len(datafile)
        
        for frame in datafile:
            totframes += 1
            image_file = os.path.join(
                    input_dir, movie_name, frame)
            featvec, isvalid = self.get_feat_per_img(image_file)
            if featvec is not None and isvalid:
                processed_frames += 1
                savefile = os.path.join(
                        trailer_feat_savedir, movie_name,
                        frame[:-3] + 'npy')
                np.save(savefile, featvec)
            else:
                error_frames += 1
        
            
        print('-'*30)
        print('no of frames that gave error: {}/{}'.format(
                error_frames, totframes))
        print('processed_frames',processed_frames)


if __name__ == "__main__":

    input_dir = './data/trailers_data/'
    movie_name = 'dark_knight_rises'
    trailer_feat_savedir = './data/trailer_feat/'
    movie_feat_savedir = './data/movie_feat/'
    extract_feat = ExtractImageFeat('resnet50')
    extract_feat.get_all_feats()