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

    def __init__(self, query_feats, query_align, search_feats, search_align,
                 windows_size=90, window_shift=10):

        self.query_feats = h5py.File(query_feats, 'r')
        self.query_align = h5py.File(query_align, 'r')

        self.search_feats = h5py.File(search_feats, 'r')
        self.search_align = h5py.File(search_align, 'r')

        # process query
        self.query_examples = []
        for ex in self.query_align:
            start = self.query_align[ex]["starts"][()][0]
            end = self.query_align[ex]["ends"][()][0]
            self.query_examples.append((ex, start, end))

        #self.query_examples = self.query_examples[:10]

        self.search_examples = []
        # process search
        for ex in self.search_align:
            seg_len = len(self.search_feats[ex][()])
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

        self.total_len = self.n_query* self.n_search

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        query_ind = item % self.n_query
        search_ind = item // self.n_query

        q_ex , q_start, q_end = self.query_examples[query_ind]
        s_ex, s_start, s_end = self.search_examples[search_ind]

        q_seg = self.query_feats[q_ex][()][q_start:q_end].astype(np.double)
        s_seg = self.search_feats[s_ex][()][s_start:s_end].astype(np.double)

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

    log.info("Evaluate on Filterbank")

    search_fn = f"/home/yushihu/Datasets/quesst2015/fbank_pitch_feats_with_cmvn.search.hdf5"
    query_fn = f"/home/yushihu/Datasets/quesst2015/fbank_pitch_feats_with_cmvn.dev-queries.hdf5"

    align_search_fn = f"/home/yushihu/Datasets/quesst2015/align.search.hdf5"
    align_query_fn = f"/home/yushihu/Datasets/quesst2015/align.dev-queries.hdf5"

    working_dir = args.dir

    dataloader = DTW_dataset(query_fn, align_query_fn, search_fn, align_search_fn,
                             windows_size=90, window_shift=10)

    cos_dist = dtw_dist_dict(dataloader)
    np.save(os.path.join(working_dir, f"dev_dist_dict.npy"), cos_dist)

    ecf_fn = f"../scoring_quesst2015/groundtruth_quesst2015_dev/quesst2015.ecf.xml"
    output_fn = os.path.join(working_dir, f"DTWSystem.stdlist.xml")

    grade_file(dist_dict=cos_dist, output_fn=output_fn, ecf_file_fn=ecf_fn,
               thres=0, score_function=default_cos_to_score)

    os.chdir(f"../scoring_quesst2015")
    os.system(f"./score-TWV-Cnxe.sh {working_dir} groundtruth_quesst2015_dev")
