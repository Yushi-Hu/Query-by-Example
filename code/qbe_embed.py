import logging as log
import os
import json
import argparse
import numpy as np
import torch
import net
import qbe_data
import shutil


def load_net(config, ckpt_dir, feat_dim, use_gpu=True, span_net=True):

    if span_net:
        acoustic_net = net.AcousticSpanRNN(config, feat_dim, use_gpu=use_gpu)

        # sketchy fix for filename
        net_view1_src = os.path.join(ckpt_dir, 'net-view1.ft.pth')
        net_view1_dst = os.path.join(ckpt_dir, 'segnet-view1.best.pth')
        net_view2_src = os.path.join(ckpt_dir, 'net-view2.ft.pth')
        net_view2_dst = os.path.join(ckpt_dir, 'segnet-view2.best.pth')
        shutil.copyfile(net_view1_src, net_view1_dst)
        shutil.copyfile(net_view2_src, net_view2_dst)

        acoustic_net.set_savepath(ckpt_dir, 'segnet')
        acoustic_net.load(tag='best')
        acoustic_net.eval()

        return acoustic_net

    else:
        acoustic_net = net.AcousticWordRNN(config, feat_dim, use_gpu=use_gpu)

        # sketchy fix for filename
        acoustic_net.set_savepath(ckpt_dir, 'net')
        acoustic_net.load(tag='best')
        acoustic_net.eval()

        return acoustic_net


def embed_searches(feats, align, ckpt_dir, config_fn, batch_size=256, span_net=True):
    query_dataset = qbe_data.SearchDataset(feats, align, stack_frames=True)
    query_loader = query_dataset.loader(batch_size=batch_size, shuffle=False)

    with open(config_fn, "r") as f:
        config = argparse.Namespace(**json.load(f))

    acoustic_net = load_net(config, ckpt_dir, query_dataset.feat_dim, use_gpu=True, span_net=span_net)

    ret = {}
    for batch in query_loader:

        ids = batch['ids']

        with torch.no_grad():
            outputs = acoustic_net.forward(batch, numpy=True)

        print(f"embed {len(ids)} segments")

        for i, k in enumerate(ids):
            if k in ret:
                print("not as expected")
            ret[k] = outputs[i]

    return ret


def embed_queries(feats, align, ckpt_dir, config_fn, batch_size=256, span_net=True):
    query_dataset = qbe_data.QueryDataset(feats, align, stack_frames=True)
    query_loader = query_dataset.loader(batch_size=batch_size, shuffle=False)

    with open(config_fn, "r") as f:
        config = argparse.Namespace(**json.load(f))

    acoustic_net = load_net(config, ckpt_dir, query_dataset.feat_dim, use_gpu=True, span_net=span_net)

    ret = {}
    for batch in query_loader:

        ids = batch['ids']
        starts = batch['starts']
        ends = batch['ends']

        with torch.no_grad():
            outputs = acoustic_net.forward(batch, numpy=True)

        print(f"embed {len(ids)} segments")

        for i, k in enumerate(ids):
            start = starts[i]
            end = ends[i]

            this_hidden = outputs[i][start:end]

            if k in ret:
                print("not as expected")
            ret[k] = this_hidden
    return ret
