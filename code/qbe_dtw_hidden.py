import numpy as np
import os
from sklearn.metrics import pairwise_distances
import xml.etree.ElementTree as ET

import sys
import functools
sys.path.append("speech_dtw")
from speech_dtw import _dtw
import logging as log
import h5py
import torch.utils.data as tud
import argparse
from qbe_query import grade_file, default_cos_to_score
import random


class DTW_dataset(tud.Dataset):

    def __init__(self, query_dict_fn, search_dict_fn,
                 windows_size=70, window_shift=5):

        self.query_feats = np.load(query_dict_fn, allow_pickle=True).item()
        self.search_feats = np.load(search_dict_fn, allow_pickle=True).item()

        # process query
        self.query_examples = list(self.query_feats.keys())

        self.search_examples = []
        # process search
        for ex in self.search_feats:
            seg_len = len(self.search_feats[ex])
            window_start = 0
            this_size = min(windows_size, seg_len)

            while window_start + this_size <= seg_len:

                window_end = window_start + this_size

                self.search_examples.append((ex, window_start, window_end))

                window_start += window_shift
                if window_end >= seg_len:
                    break

        random.shuffle(self.query_examples)
        random.shuffle(self.search_examples)

        self.n_query = len(self.query_examples)
        self.n_search = len(self.search_examples)

        print(self.n_query, self.n_search)

        self.total_len = self.n_query * self.n_search

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        query_ind = item % self.n_query
        search_ind = item // self.n_query

        q_ex = self.query_examples[query_ind]
        s_ex, s_start, s_end = self.search_examples[query_ind]

        q_seg = self.query_feats[q_ex].astype(np.double)
        s_seg = self.search_feats[s_ex][s_start:s_end].astype(np.double)

        return {"query_id": q_ex, "query_seg": q_seg,
                "search_id": s_ex, "search_seg": s_seg}


def dtw_dist_dict(dataloader):

    count = 0
    dist_func = functools.partial(_dtw.multivariate_dtw_cost_cosine,
                                  dur_normalize=True)
    log.info(f"need to evaluate {len(dataloader)} pairs")
    dist_dict = {}
    for i in range(dataloader.__len__()):
        seg_pair = dataloader.__getitem__(i)
        q_ex = seg_pair["query_id"]
        q_seg = seg_pair["query_seg"]
        s_ex = seg_pair["search_id"]
        s_seg = seg_pair["search_seg"]
        this_dist = dist_func(q_seg, s_seg)
        count += 1
        if count % 10000 == 0:
            log.info(f"evaluated {count} pairs")

        if (q_ex, s_ex) in dist_dict:
            dist_dict[(q_ex, s_ex)] = min(dist_dict[(q_ex, s_ex)], this_dist)
        else:
            dist_dict[(q_ex, s_ex)] = this_dist
    return dist_dict

if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="working directory")
    args = parser.parse_args()

    working_dir = args.dir
    hidden_dir = os.path.join(working_dir, "2015results")

    dev_hidden_fn = os.path.join(hidden_dir, f'dev_query_hidden.npy')
    search_hidden_fn = os.path.join(hidden_dir, f'search_hidden.npy')

    dataloader = DTW_dataset(dev_hidden_fn, search_hidden_fn,
                             windows_size=45, window_shift=5)

    working_dir = os.path.join(working_dir, "DTW_results")
    os.makedirs(working_dir, exist_ok=True)

    cos_dist = dtw_dist_dict(dataloader)
    np.save(os.path.join(working_dir, f"dev_dist_dict.npy"), cos_dist)

    ecf_fn = f"../scoring_quesst2015/groundtruth_quesst2015_dev/quesst2015.ecf.xml"
    output_fn = os.path.join(working_dir, f"DTWSystem.stdlist.xml")

    grade_file(dist_dict=cos_dist, output_fn=output_fn, ecf_file_fn=ecf_fn,
               thres=0, score_function=default_cos_to_score)

    os.chdir(f"../scoring_quesst2015")
    os.system(f"./score-TWV-Cnxe.sh {working_dir} groundtruth_quesst2015_dev")
