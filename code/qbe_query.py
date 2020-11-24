import numpy as np
import os
import xml.etree.ElementTree as ET
import pandas as pd
from scipy.spatial.distance import cdist
import logging

def hidden_to_emb(hidden, start=None, end=None, mode='mean'):
    if mode == 'mean':
        if start is None and end is None:
            return hidden.mean(0)
        else:
            return hidden[start:end].mean(0)

    elif mode == 'concat':
        seq_len = hidden.shape[0]
        hidden = hidden.reshape((seq_len, 2, -1))

        if start is None and end is None:
            start = 0
            end = 0  # a trick. 0-1 = -1

        return np.concatenate((hidden[end - 1, 0], hidden[start, 1]), axis=-1)

    else:
        raise ValueError("mode should be mean or concat")


def sample_len(start=12, mid=30, large=120, small_inc=3, large_inc=6):
    return list(range(start, mid, small_inc)) + list(range(mid, large + large_inc, large_inc))


def build_search_index(search_hidden_fn, windows, increment, mode):
    search_dict = np.load(search_hidden_fn, allow_pickle=True).item()
    sks = search_dict.keys()
    df_dict = {}
    df_columns = ['emb', 'search_id']
    emb_dict = {}
    sid_dict = {}

    # make data frame to store embeddings
    for s_len in windows:
        df_dict[s_len] = pd.DataFrame(columns=df_columns)
        emb_dict[s_len] = []
        sid_dict[s_len] = []

    segment_count= 0

    for sk in sks:
        #print(f"now processing {sk}", flush=True)

        hidden = search_dict[sk]

        search_len = hidden.shape[0]

        for start in range(0, search_len, increment):
            for s_len in windows:

                end = start + s_len
                if end >= search_len:
                    continue

                this_emb = hidden_to_emb(hidden, start, end, mode=mode)

                emb_dict[s_len].append(this_emb)
                sid_dict[s_len].append(sk)

                segment_count += 1

                # this_row = {'emb': this_emb, 'search_id': sk}
                # df_dict[s_len].append(this_row, ignore_index=True)

    logging.info(f"in total {segment_count} segments")
    # stack all embeddings for parallel computation. The index is the length of segments
    all_emb_df = pd.DataFrame(columns=['embs', 'search_ids'])

    for s_len in windows:
        """
        embs = []
        ids = []
        for index, row in df_dict[s_len].iterrows():
            embs.append(row['emb'])
            ids.append(row['id'])
        """
        all_emb_df.loc[s_len] = {'embs': np.vstack(emb_dict[s_len]), 'search_ids': sid_dict[s_len]}
        # save space
        del df_dict[s_len]
        del emb_dict[s_len]
        print(f"finish processing {s_len}", flush=True)

    print('finish building index')
    return sks, all_emb_df


def cos_dist_dict(query_hidden_fn, search_hidden_fn, windows, increment=5,
                  mode="mean", min_thres=2 / 3, max_thres=4 / 3):
    query_dict = np.load(query_hidden_fn, allow_pickle=True).item()
    qks = query_dict.keys()

    dist_dict = {}
    max_dist = 0

    sks, all_emb_df = build_search_index(search_hidden_fn, windows, increment, mode)

    all_n_comp = []

    for qk in qks:

        n_comp = 0
        hidden = query_dict[qk]
        query_len = hidden.shape[0]
        query_emb = hidden_to_emb(hidden, mode=mode).reshape(1, -1)

        for s_len in windows:
            if min_thres * query_len <= s_len <= max_thres * query_len:
                this_embs = all_emb_df.loc[s_len]['embs']
                this_ids = all_emb_df.loc[s_len]['search_ids']
                n_comp += len(this_ids)

                #dist = my_cos_cdist(this_embs, query_emb) / 2
                dist = cdist(this_embs, query_emb, metric="cosine")
                for i, name in enumerate(this_ids):
                    this_dist = dist[i].item()
                    max_dist = max(max_dist, this_dist)

                    if (qk, name) not in dist_dict:
                        dist_dict[(qk, name)] = this_dist
                    else:
                        if this_dist < dist_dict[(qk, name)]:
                            dist_dict[(qk, name)] = this_dist
        logging.info(f"querying {qk} compares {n_comp} segments")
        all_n_comp.append(n_comp)

    logging.info(f"on average compare {sum(all_n_comp)/len(qks)} segments")
    return dist_dict


def default_cos_to_score(s):
    return 10 * (1.5 - s) - 7.5


def grade_file(dist_dict, output_fn, ecf_file_fn, thres=0, score_function=None):
    # get query and search keys
    query_set = set()
    search_set = set()
    for qk, sk in dist_dict.keys():
        query_set.add(qk)
        search_set.add(sk)

    query_list = sorted(list(query_set))
    search_list = sorted(list(search_set))

    # get query durations
    ecf_tree = ET.parse(ecf_file_fn)
    root = ecf_tree.getroot()
    audio_time_dict = {}
    for child in root:
        child = child.attrib
        fn = child['audio_filename']
        _, fn = os.path.split(fn)
        fn = os.path.splitext(fn)[0]
        dur = child['dur']
        audio_time_dict[fn] = dur

    # for grading script
    result_heading = """<?xml version="1.0" encoding="UTF-8"?>
    <stdlist termlist_filename="Query.tlist.xml" indexing_time="1.00" language="mul" index_size="1" system_id="AWE system">
    """
    result_ending = "</stdlist>\n"

    term_heading = """<detected_termlist termid="{}" term_search_time="2.00" oov_term_count="1">
    """
    term_ending = "</detected_termlist>\n"
    one_query = """<term file="{}" channel="1" tbeg="0" dur="{}" score="{}" decision="{}"/>
    """

    with open(output_fn, 'w') as f:
        f.write(result_heading)

        for q in query_list:
            f.write(term_heading.format(q))

            for s in search_list:

                if (q, s) in dist_dict:
                    score = dist_dict[(q, s)]

                    result = 'NO'

                    if score_function is not None:
                        score = score_function(score)

                    if score > thres:
                        result = 'YES'

                    f.write(one_query.format(s, audio_time_dict[s], score, result))

            f.write(term_ending)
        f.write(result_ending)

    return
