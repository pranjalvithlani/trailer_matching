# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:43:36 2020

@author: Pranjal Vithlani
"""

import torch
import os
import numpy as np
import datetime


def calc_error_for_retrieval(val1, val2, lossfunc):
    '''
    INPUT:
    val1 --> B x 1 X dim - B - batchsize
    val2 --> 1 x N x dim - N - no of images in test set
    RETURN:
         --> B X N - error
    '''
    
    err = val1 - val2
    
    # for some reason during eval, gpu gives OOM here. This is temp fix
    torch.cuda.empty_cache()
    if lossfunc == 'mse':
        err = torch.mean(torch.pow(err, 2), dim=-1, keepdim=True)
    elif lossfunc == 'mae':
        err = torch.mean(val1, dim=-1, keepdim=True)
    else:
        raise ValueError('only mse, mae and order supported now')
    return err



def retrieve_images(pred_feats, frame_feats):
    '''
    INPUT:
    pred_feats  --> B x 1 X dim - B - batchsize
    frame_feats --> 1 x N x dim - N - no of images in test set
    
    RETURN:
    output      --> B x 1 x dim
    topkidx     --> B x 1 x 1
    '''
    
    lossfunc = 'mae'
    
    # calc error -> B x N
    error = calc_error_for_retrieval(pred_feats, frame_feats, lossfunc).squeeze(
                    -1)
    
    
    sortedvals, sortedidx = torch.sort(error, dim=1, descending=False)
    
    # retrieve topk
    topkidx = sortedidx[:, :1]
    
    # get top1 indices as a tensor
    top1 = topkidx[:, 0].unsqueeze(1)
    # get top1 GT FEATS to be used as input in next step
    output = [frame_feats[0, i[0], :].unsqueeze(0) for i in top1]
    output = torch.cat(output, dim=0).unsqueeze(1)
    
    return output, topkidx.unsqueeze(1)


def main():
    trailer_load_start = datetime.datetime.now()
    
    trailer_feat_loc = './data/trailer_feat/dark_knight_rises/'
    trailer_feat_ = list()
    for i in os.listdir(trailer_feat_loc):
        a = np.load(trailer_feat_loc+i)
        trailer_feat_.append(a)
    # trailer_feat = np.asarray(trailer_feat_)
    # del trailer_feat_
    trailer_feat = torch.as_tensor(trailer_feat_).cuda()
    #trailer_feat = torch.cat(trailer_feat_,0)
    trailer_feat = trailer_feat.unsqueeze(1)
    print(trailer_feat.shape)
    
    trailer_load_end = datetime.datetime.now()
    print((trailer_load_end - trailer_load_start).total_seconds())
    
    movie_load_start = datetime.datetime.now()
    
    movie_feat_loc = './data/movie_feat/dark_knight_rises_movie/'
    movie_feat_ = list()
    for i in os.listdir(movie_feat_loc):
        a = np.load(movie_feat_loc+i)
        movie_feat_.append(a)
    # movie_feat = np.asarray(movie_feat_)
    # del movie_feat_
    movie_feat = torch.as_tensor(movie_feat_).cuda()
    #movie_feat = torch.cat(movie_feat_[:100000],0)
    movie_feat = movie_feat.unsqueeze(0)
    print(movie_feat.shape)
    
    movie_load_end = datetime.datetime.now()
    print((movie_load_end - movie_load_start).total_seconds())
    
    retrieve_start = datetime.datetime.now()
    ans_list = list()
    for i in trailer_feat:
        ot,idx = retrieve_images(i.unsqueeze(1),movie_feat)
        ans_list.append(idx.item())
    
    retrieve_end = datetime.datetime.now()
    
    
    print(ans_list)
    savefile = os.path.join('ans_list.npy')
    np.save(savefile, ans_list)
    print((retrieve_end - retrieve_start).total_seconds())
    
if __name__ == "__main__":
    main()