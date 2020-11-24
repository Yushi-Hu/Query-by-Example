import logging as log
import h5py
import json
import collections
import numpy as np
import torch
import torch.utils.data as tud

import utils.stateful_dataset
import utils.speech_utils


# help to unify all the vocabs
class CountDict:
    def __init__(self):
        self.dict = {}

    def insert(self, x):
        if x in self.dict:
            self.dict[x] += 1
        else:
            self.dict[x] = 1
        return

    def elt_set(self):
        return set(self.dict.keys())

    def idx_dict(self):
        sorted_d = sorted(self.dict.items(), key=lambda x: x[1])
        sorted_d.reverse()
        idx = 0
        result_dict = {}
        for k, v in sorted_d:
            result_dict[k] = idx
            idx += 1
        return result_dict


def use_spec_aug(feat, F=27, T=100, m_F=1, m_T=1):
    n, d = feat.shape
    T = max(min(n // 4, T), 2)
    F = max(min(d // 4, F), 2)
    if m_F > 0:
        for i in range(m_F):  # freq masks
            f = np.random.randint(1, F)
            f0 = np.random.randint(0, d - f + 1)
            feat.T[f0:f0 + f] = 0.
    if m_T > 0:
        for i in range(m_T):  # time masks
            t = np.random.randint(1, T)
            t0 = np.random.randint(0, n - t + 1)
            feat[t0:t0 + t] = 0.
    return feat


def combine_subwords_to_ids(vocab_fns, subwords):
    vocab_dict = CountDict()
    for vocab_fn in vocab_fns:
        with open(vocab_fn, "r") as f:
            vocab = json.load(f)
        for k, v in vocab[f"{subwords}_to_ids"].items():
            vocab_dict.insert(k)
    return vocab_dict.idx_dict()


class MultilangDataset(utils.stateful_dataset.StatefulDataset):

    def __init__(self, feats_fns, align_fns, vocab_fns, subwords, subwords_to_ids,
                 min_word_dur=6, min_seg_dur=6, stack_frames=False,
                 batch_size=1, shuffle=False, variable=False, cache=None, spec_aug=False):

        super().__init__()

        if cache is not None:
            feats_fns = [cache(fn) for fn in feats_fns]
            align_fns = [cache(fn) for fn in align_fns]

        self.stack_frames = stack_frames
        self.spec_aug = spec_aug
        self.min_word_dur = min_word_dur

        # number of langauges
        self.n_languages = len(feats_fns)
        log.info(f"Using {self.n_languages}languages")

        log.info(f"Using {feats_fns}; stacked={stack_frames}")
        featss = [h5py.File(fn, "r") for fn in feats_fns]

        log.info(f"Using {align_fns}")
        aligns = [h5py.File(fn, "r") for fn in align_fns]

        # using different embeddings
        log.info(f"Using {vocab_fns}")

        self.subwords_to_ids = subwords_to_ids
        self.n_subwords = len(self.subwords_to_ids)
        self.min_seg_dur = min_seg_dur

        log.info(f"Using {self.n_subwords} tokens in subwords")

        # all training_data
        self.langs_data = []

        # training samples
        for lang_id, vocab_fn in enumerate(vocab_fns):

            # process everything for a single language with index lang_id
            with open(vocab_fn, "r") as f:
                vocab = json.load(f)
            lang_dict = {"words_to_ids": vocab["words_to_ids"], "word_to_subwords": vocab[f"word_to_{subwords}"]}

            # counting words
            align = aligns[lang_id]
            feats = featss[lang_id]

            lang_dict['feats'] = feats
            lang_dict['align'] = align
            lang_dict['ids_to_words'] = {v: k for k, v in lang_dict["words_to_ids"].items()}

            # total training frames
            total_frame = 0.0
            total_train_instances = 0

            # train examples of this language
            examples = {}
            for cs in align:

                # a single utterance
                for uid, g in align[cs].items():

                    ind = []  # legal word indices in an utterance. 0, 1, 2, ...
                    word_count_in_utt = 0
                    utt_len = len(feats[cs][uid]["feats"][()])

                    if utt_len < min_seg_dur:
                        continue

                    words = g["words"][()]
                    durs = g["ends"][()] - g["starts"][()] + 1

                    for i, (word, dur) in enumerate(zip(words, durs)):

                        if word[0] == '<' and word[-1] == '>':
                            continue
                        word_count_in_utt += 1
                        ind.append(i)  # just remove the unknown words

                    if word_count_in_utt == 0:
                        continue

                    examples[(cs, uid, tuple(ind))] = {
                        "frames": utt_len // 2 if stack_frames else utt_len,
                        "words": word_count_in_utt
                    }

                    total_frame += utt_len
                    total_train_instances += word_count_in_utt

            log.info(f"total train data time {total_frame / 60 / 100} minutes")
            log.info(f"total train data instances {total_train_instances}")

            # sort examples by the length
            examples = sorted(examples.items(), key=lambda x: x[1]["frames"], reverse=True)

            lang_dict['examples'] = examples

            self.langs_data.append(lang_dict)

        # all examples
        all_examples = []

        for i in range(self.n_languages):
            all_examples.append(self.langs_data[i]['examples'])

        # get feature dimension
        def get_feat_dim():
            for lang_dict in self.langs_data:
                feats = lang_dict['feats']
                for cs in feats:
                    for uid, g in feats[cs].items():
                        seg = g["feats"][()]
                        # seg = utils.speech_utils.add_deltas(seg)
                        if self.stack_frames:
                            seg = utils.speech_utils.stack(seg)
                        return seg.shape[1]

        self.feat_dim = get_feat_dim()

        self.len = len(all_examples)

        # dataloader
        batch_sampler = utils.stateful_dataset.MultilangPackedBatchSampler(
            all_examples, batch_size=batch_size, shuffle=shuffle, variable=variable)

        loader = tud.DataLoader(self,
                                batch_sampler=batch_sampler,
                                collate_fn=self.collate_fn,
                                num_workers=1)

        self.loader = loader


    @property
    def num_subwords(self):
        return self.n_subwords

    @property
    def unify_subwords_to_ids(self):
        return self.subwords_to_ids

    # random partition starts and ends
    def partition(self, starts, ends, seg_len, word_ids):

        # number of words in the segment
        n_word = len(starts)

        # start and end of partition
        seg_start, seg_end = [0, starts[0]], [ends[-1], seg_len]
        start_end_idx = np.random.choice(2, 2)
        seg_start, seg_end = seg_start[start_end_idx[0]], seg_end[start_end_idx[1]]

        n_divs = 0
        if n_word >= 2:
            n_divs = np.random.choice(n_word // 2, 1).item()  # how many divides in the span

        div_pts = []
        div_pt_idxs = []

        if n_divs != 0:
            div_pt_idxs = np.random.choice(n_word - 1, size=n_divs, replace=False)
            div_flags = np.random.choice(2, n_divs)
            div_pts = [starts[idx + 1] for i, idx in enumerate(div_pt_idxs) if div_flags[i] == 0] + \
                      [ends[idx] for i, idx in enumerate(div_pt_idxs) if div_flags[i] == 1]
            div_pts = sorted(div_pts)
            div_pt_idxs += 1

        div_pt_idxs = sorted(list(div_pt_idxs)) + [n_word]

        # get word ids that are in each span
        segs_word_ids = []
        segs_word_ids.append(list(range(div_pt_idxs[0])))
        for i in range(n_divs):
            segs_word_ids.append(list(range(div_pt_idxs[i], div_pt_idxs[i + 1])))

        span_starts = [seg_start] + div_pts
        span_ends = div_pts + [seg_end]

        span_starts_final = []
        span_ends_final = []
        segs_word_ids_final = []

        for this_start, this_end, this_ids in zip(span_starts, span_ends, segs_word_ids):
            if this_end - this_start > self.min_seg_dur:
                span_starts_final.append(this_start)
                span_ends_final.append(this_end)
                segs_word_ids_final.append(this_ids)

        span_word_ids = [[word_ids[i] for i in this_span] for this_span in segs_word_ids_final]

        return span_starts_final, span_ends_final, span_word_ids

    def __getitem__(self, ex):
        # return everything needed for an utterance example
        lang_id, cs, uid, ind = ex
        lang_dict = self.langs_data[lang_id]
        align = lang_dict['align']
        feats = lang_dict['feats']
        words_to_ids = lang_dict['words_to_ids']
        word_to_subwords = lang_dict['word_to_subwords']

        seg = feats[cs][uid]["feats"][()]
        starts = [align[cs][uid]["starts"][()][i] for i in ind]
        ends = [align[cs][uid]["ends"][()][i] for i in ind]

        # spec augment
        if self.spec_aug:
            min_word_frame_len = np.amin(np.array(ends) - np.array(starts))
            seg = use_spec_aug(seg, T=min_word_frame_len // 2)  # mask at most half of the shortest word

        # seg = utils.speech_utils.add_deltas(seg)
        if self.stack_frames:
            seg = utils.speech_utils.stack(seg)
            starts = [start // 2 for start in starts]
            ends = [end // 2 for end in ends]

        if len(seg) == 0:
            return None

        words = [align[cs][uid]["words"][()][i] for i in ind]
        word_ids = [words_to_ids[w] for w in words]

        # get subword sequence for each single word
        word_subword_seqs = [word_to_subwords[w][0] for w in words]
        word_subword_seq_ids = [[self.subwords_to_ids[p] + 1 for p in s] for s in word_subword_seqs]
        word_subword_seq_lens = [len(s) for s in word_subword_seqs]

        if len(word_subword_seq_ids) == 0:
            return None

        span_starts, span_ends, span_word_ids = self.partition(starts, ends, len(seg), word_ids)

        return {"seg": seg,
                # about words
                "word_ids": word_ids, "word_starts": starts, "word_ends": ends,
                "seq_ids": word_subword_seq_ids, "seq_lens": word_subword_seq_lens,
                # about spans
                "span_starts": span_starts, "span_ends": span_ends,
                "span_word_ids": span_word_ids}

    def collate_fn(self, batch):

        batch_size = len(batch)
        durs = torch.LongTensor([len(ex["seg"]) for ex in batch])  # length of each utterance in the batch
        segs = torch.zeros(len(durs), max(durs), self.feat_dim)
        word_starts, word_ends, utt_words_ids = [], [], []
        span_starts, span_ends, span_word_ids = [], [], []
        n_word = 0
        n_span = 0
        for i, ex in enumerate(batch):
            # put features in span
            segs[i, :durs[i]] = torch.from_numpy(ex["seg"])
            # about words
            word_starts.append(ex["word_starts"])
            word_ends.append(ex["word_ends"])
            n_word += len(ex["word_starts"])

            utt_words_ids.append(ex["word_ids"])
            # about spans
            span_starts.append(ex["span_starts"])
            span_ends.append(ex["span_ends"])
            n_span += len(ex["span_starts"])

            span_word_ids.append(ex["span_word_ids"])

        if n_span == 0 or n_word == 0:
            return None

        # for word discrimination usage, get the ids of word segment with length larger than threshold
        legal_word_inds = []
        this_word_ind = 0
        for i in range(batch_size):
            for this_word_start, this_word_end in zip(word_starts[i], word_ends[i]):
                if this_word_end - this_word_start > self.min_word_dur:
                    legal_word_inds.append(this_word_ind)
                this_word_ind += 1

        legal_word_inds = np.array(legal_word_inds, dtype=int)

        ##############################
        # Below are about written side
        ##############################

        # written word embeddings

        word_view2_lens = torch.cat([torch.tensor(ex["seq_lens"]) for ex in batch]).type(torch.long)
        word_view2 = torch.zeros(len(word_view2_lens), max(word_view2_lens), dtype=torch.long)

        j = 0
        for i, ex in enumerate(batch):
            for this_seq, this_len in zip(ex["seq_ids"], ex["seq_lens"]):
                word_view2[j, :this_len] = torch.tensor(this_seq, dtype=torch.long)
                j += 1

        word_ids = np.hstack([np.array(utt_words_ids[i]) for i in range(batch_size)]).astype(np.int32)
        word_uids, word_ind, word_inv_ind = np.unique(
            word_ids, return_index=True, return_inverse=True)

        word_uid_to_word_inv = {uid: i for i, uid in enumerate(word_uids)}

        # written span embeddings

        span_view2_lens = []
        span_word_invs_view2 = []  # let network feed in word_invs for convenience

        for i, this_utt in enumerate(span_word_ids):
            for this_span in this_utt:
                span_view2_lens.append(len(this_span))
                span_word_invs_view2.append([word_uid_to_word_inv[uid] for uid in this_span])

        span_view2_lens = torch.tensor(span_view2_lens).type(torch.long)

        # get span ids
        span_set = set()
        for utt in span_word_ids:
            for span in utt:
                span_set.add(tuple(span))

        span_to_span_uids = {span: uid for uid, span in enumerate(span_set)}

        # get the span_ids
        span_ids = []
        flatten_span_word_ids = []
        for utt in span_word_ids:
            for span in utt:
                span_ids.append(span_to_span_uids[tuple(span)])
                flatten_span_word_ids.append(span)

        span_ids = np.array(span_ids).astype(np.int32)
        span_uids, span_ind, span_inv_ind = np.unique(
            span_ids, return_index=True, return_inverse=True)

        ################################
        # Below are for prediction loss
        ################################

        # check if q is a continuous sublist of unique spans
        def chk_sublist(q, s):
            q, s = list(q), list(s)
            len_diff = len(s) - len(q)
            if len_diff < 0 or (q[0] not in s):
                return False
            for i in range(len_diff + 1):
                if s[i:i + len(q)] == q:
                    return True
            return False

        # return a matrix to show if a span is in a segment
        # dimension (batch_size, len(span_uids))
        # if there is a hit, then the value is 1. else 0

        utt_uspan_gold = torch.zeros(batch_size, len(span_uids)).type(torch.long)

        for utt_id in range(batch_size):
            word_in_utt = utt_words_ids[utt_id]
            for s, u in span_to_span_uids.items():
                if chk_sublist(s, word_in_utt):
                    utt_uspan_gold[utt_id, u] = 1

        utt_allspan_gold = torch.zeros(batch_size, len(span_inv_ind)).type(torch.long)

        for utt_id in range(batch_size):
            utt_allspan_gold[utt_id, :] = utt_uspan_gold[utt_id][span_inv_ind]

        return {
            "view1": segs, "view1_lens": durs,
            "word_starts": word_starts, "word_ends": word_ends,
            "word_view2": word_view2[word_ind], "word_view2_lens": word_view2_lens[word_ind],
            "word_uids": word_uids, "word_inv": torch.from_numpy(word_inv_ind), "legal_word_inds": legal_word_inds,
            "span_starts": span_starts, "span_ends": span_ends,
            "span_view2_lens": [span_view2_lens[i] for i in span_ind],
            "span_word_invs_view2": [span_word_invs_view2[i] for i in span_ind],
            "span_uids": span_uids, "span_inv": torch.from_numpy(span_inv_ind),
            "span_word_ids": [flatten_span_word_ids[i] for i in span_ind],
            "utt_uspan_gold": utt_uspan_gold, "utt_allspan_gold": utt_allspan_gold
        }
