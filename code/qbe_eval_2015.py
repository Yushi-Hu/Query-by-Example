import logging as log
import os
import json
import argparse
import random
import numpy as np
import torch
import logging

from qbe_embed import embed_queries, embed_searches
from qbe_query import cos_dist_dict, grade_file, sample_len, default_cos_to_score

if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="working directory")
    parser.add_argument("--mode", default='concat', help="mean or concat")
    parser.add_argument("--inc", type=int, default=5, help="frame resolution")
    parser.add_argument("--emb_level", choices=['span', 'word'], default='span', help="ASE or AWE")
    args = parser.parse_args()

    search_fn = f"/home/yushihu/Datasets/quesst2015/fbank_pitch_feats_with_cmvn.search.hdf5"
    query_fn = f"/home/yushihu/Datasets/quesst2015/fbank_pitch_feats_with_cmvn.dev-queries.hdf5"

    align_search_fn = f"/home/yushihu/Datasets/quesst2015/align.search.hdf5"
    align_query_fn = f"/home/yushihu/Datasets/quesst2015/align.dev-queries.hdf5"

    test_query_fn = f"/home/yushihu/Datasets/quesst2015/fbank_pitch_feats_with_cmvn.eval-queries.hdf5"
    test_align_query_fn = f"/home/yushihu/Datasets/quesst2015/align.eval-queries.hdf5"

    working_dir = args.dir
    mode = args.mode
    increment = args.inc
    span_net = True
    if args.emb_level == 'word':
        span_net = False

    ckpt_dir = os.path.join(working_dir, "save/")
    config_fn = os.path.join(working_dir, "train_config.json")

    # change working directory to save outputs
    working_dir = os.path.join(working_dir, "2015results")
    os.makedirs(working_dir, exist_ok=True)

    logging.info(f"Start building index")

    # embed sequences
    """
    # dev set
    dev_hidden_fn = os.path.join(working_dir, f'dev_query_hidden.npy')
    query_dict = embed_queries(feats=query_fn, align=align_query_fn,
                               ckpt_dir=ckpt_dir, config_fn=config_fn,
                               span_net=span_net)
    np.save(dev_hidden_fn, query_dict)
    del query_dict

    # search set
    search_hidden_fn = os.path.join(working_dir, f'search_hidden.npy')
    search_dict = embed_searches(feats=search_fn, align=align_search_fn,
                                 ckpt_dir=ckpt_dir, config_fn=config_fn,
                                 span_net=span_net)
    np.save(search_hidden_fn, search_dict)
    del search_dict

    # eval set
    eval_hidden_fn = os.path.join(working_dir, f'eval_query_hidden.npy')
    query_dict = embed_queries(feats=test_query_fn, align=test_align_query_fn,
                               ckpt_dir=ckpt_dir, config_fn=config_fn,
                               span_net=span_net)
    np.save(eval_hidden_fn, query_dict)
    del query_dict
    # """

    dev_hidden_fn = os.path.join(working_dir, f'dev_query_hidden.npy')
    search_hidden_fn = os.path.join(working_dir, f'search_hidden.npy')
    eval_hidden_fn = os.path.join(working_dir, f'eval_query_hidden.npy')

    logging.info("finish building index")

    # get cosine distance
    cos_dist = cos_dist_dict(eval_hidden_fn, search_hidden_fn, windows=sample_len(), increment=increment, mode=mode)
    np.save(os.path.join(working_dir, f"eval_dist_dict.npy"), cos_dist)

    # cos_dist = np.load(os.path.join(working_dir, "dev_dist_dict.npy"), allow_pickle=True).item()

    ecf_fn = f"../scoring_quesst2015/groundtruth_quesst2015_eval/quesst2015.ecf.xml"
    working_dir = os.path.join(working_dir, 'eval')
    os.makedirs(working_dir, exist_ok=True)
    output_fn = os.path.join(working_dir, f"AWESystem.stdlist.xml")

    grade_file(dist_dict=cos_dist, output_fn=output_fn, ecf_file_fn=ecf_fn,
               thres=0, score_function=default_cos_to_score)

    os.chdir(f"../scoring_quesst2015")
    os.system(f"./score-TWV-Cnxe.sh {working_dir} groundtruth_quesst2015_eval -7.5")