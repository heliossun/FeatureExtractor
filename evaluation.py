from __future__ import print_function
import os
import pickle
import torch
import numpy
import time
import numpy as np


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    for i, (images, captions, lengths, ids, caption_labels, caption_masks) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb, GCN_img_emd = model.forward_emb(images, captions, lengths,
                                                          volatile=True)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        del images, captions

    return img_embs, cap_embs


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    if npts is None:
        npts = images.shape[0]

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[index].reshape(1, images.shape[1])

        # Compute scores
        d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]

        # print(inds)
        # Score
        rank = 1e20
        where = numpy.where(inds == index)
        # print("where:  ", where)
        tmp = where[0][0]
        if tmp < rank:
            rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
    # print("ranks: ",ranks)
    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    if npts is None:
        npts = images.shape[0]

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):
        cap = captions[index]
        # Compute scores
        d = numpy.dot(cap, images.T)
        inds = numpy.argsort(d)[::-1]
        ranks[index] = numpy.where(inds == index)[0][0]
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
