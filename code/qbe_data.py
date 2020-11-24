import logging as log
import h5py
import json
import collections
import numpy as np
import torch
import torch.utils.data as tud
import utils.stateful_dataset
import utils.speech_utils
import random


class SearchDataset(tud.Dataset):

    def __init__(self, feats_fn, align_fn, stack_frames=False):

        super().__init__()
        self.stack_frames = stack_frames

        log.info(f"Using {feats_fn}; stacked={stack_frames}")
        self.feats = h5py.File(feats_fn, "r")

        log.info(f"Using {align_fn}")
        self.align = h5py.File(align_fn, "r")

        self.examples = []

        for ex in self.align:
            self.examples.append(ex)

        # get feature dimension
        seg = self.feats[self.examples[0]][()]
        # seg = utils.speech_utils.add_deltas(seg)
        if self.stack_frames:
            seg = utils.speech_utils.stack(seg)
        self.feat_dim = seg.shape[1]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        ex = self.examples[idx]

        seg = self.feats[ex][()]

        # seg = utils.speech_utils.add_deltas(seg)

        if self.stack_frames:
            seg = utils.speech_utils.stack(seg)

        return {"seg": seg, "query": ex}

    def collate_fn(self, batch):

        batch_size = 0
        max_seg_dur = 0

        ids = []
        for ex in batch:
            if ex is not None:
                max_seg_dur = max(max_seg_dur, len(ex["seg"]))
                ids.append(ex["query"])
                batch_size += 1

        feats = torch.zeros(batch_size, max_seg_dur, self.feat_dim)
        feat_lens = torch.zeros(batch_size, dtype=torch.long)

        i = 0
        for ex in batch:
            if ex is not None:
                seg = ex["seg"]
                feats[i, :len(seg)] = torch.from_numpy(seg)
                feat_lens[i] = len(seg)
                i += 1

        return {
            "view1": feats, "view1_lens": feat_lens,
            "ids": ids
        }

    def loader(self, batch_size=1, shuffle=False):
        return tud.DataLoader(self,
                              batch_size=batch_size,
                              collate_fn=self.collate_fn,
                              shuffle=shuffle,
                              num_workers=0)


class FineTuneQueryDataset(tud.Dataset):

    def __init__(self, feats_fn, align_fn, qks, stack_frames=False):

        super().__init__()
        self.stack_frames = stack_frames

        log.info(f"Using {feats_fn}; stacked={stack_frames}")
        self.feats = h5py.File(feats_fn, "r")

        log.info(f"Using {align_fn}")
        self.align = h5py.File(align_fn, "r")

        self.examples = []

        for ex in qks:
            starts = self.align[ex]["starts"][()]
            ends = self.align[ex]["ends"][()]

            for start, end in zip(starts, ends):
                self.examples.append((ex, start, end))

        # get feature dimension

        seg = self.feats[self.examples[0][0]][()]
        # seg = utils.speech_utils.add_deltas(seg)
        if self.stack_frames:
            seg = utils.speech_utils.stack(seg)
        self.feat_dim = seg.shape[1]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        ex, start, end = self.examples[idx]

        seg = self.feats[ex][()]

        if len(seg) == 0:
            return None

        # seg = utils.speech_utils.add_deltas(seg)

        if self.stack_frames:
            seg = utils.speech_utils.stack(seg)

        if self.stack_frames:
            start = start // 2
            end = end // 2

        return {"seg": seg, "query": ex, "start": start, "end": end}

    def collate_fn(self, batch):

        batch_size = 0
        max_seg_dur = 0

        ids = []
        starts = []
        ends = []
        for ex in batch:
            if ex is not None:
                max_seg_dur = max(max_seg_dur, len(ex["seg"]))
                ids.append(ex["query"])
                starts.append(ex["start"])
                ends.append(ex["end"])
                batch_size += 1

        feats = torch.zeros(batch_size, max_seg_dur, self.feat_dim)
        feat_lens = torch.zeros(batch_size, dtype=torch.long)

        i = 0
        for ex in batch:
            if ex is not None:
                seg = ex["seg"]
                feats[i, :len(seg)] = torch.from_numpy(seg)
                feat_lens[i] = len(seg)
                i += 1

        return {
            "view1": feats, "view1_lens": feat_lens,
            "ids": ids, "starts": starts, "ends": ends
        }

    def loader(self, batch_size=1, shuffle=False):
        return tud.DataLoader(self,
                              batch_size=batch_size,
                              collate_fn=self.collate_fn,
                              shuffle=shuffle,
                              num_workers=0)


class QueryDataset(tud.Dataset):

    def __init__(self, feats_fn, align_fn, stack_frames=False):

        super().__init__()
        self.stack_frames = stack_frames

        log.info(f"Using {feats_fn}; stacked={stack_frames}")
        self.feats = h5py.File(feats_fn, "r")

        log.info(f"Using {align_fn}")
        self.align = h5py.File(align_fn, "r")

        self.examples = []

        for ex in self.align:
            starts = self.align[ex]["starts"][()]
            ends = self.align[ex]["ends"][()]
            # some end-start < 10

            for start, end in zip(starts, ends):
                self.examples.append((ex, start, end))

        # get feature dimension
        seg = self.feats[self.examples[0][0]][()]
        # seg = utils.speech_utils.add_deltas(seg)
        if self.stack_frames:
            seg = utils.speech_utils.stack(seg)
        self.feat_dim = seg.shape[1]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        ex, start, end = self.examples[idx]

        seg = self.feats[ex][()]

        if len(seg) == 0:
            return None

        # seg = utils.speech_utils.add_deltas(seg)

        if self.stack_frames:
            seg = utils.speech_utils.stack(seg)

        if self.stack_frames:
            start = start // 2
            end = end // 2

        return {"seg": seg, "query": ex, "start": start, "end": end}

    def collate_fn(self, batch):

        batch_size = 0
        max_seg_dur = 0

        ids = []
        starts = []
        ends = []
        for ex in batch:
            if ex is not None:
                max_seg_dur = max(max_seg_dur, len(ex["seg"]))
                ids.append(ex["query"])
                starts.append(ex["start"])
                ends.append(ex["end"])
                batch_size += 1

        feats = torch.zeros(batch_size, max_seg_dur, self.feat_dim)
        feat_lens = torch.zeros(batch_size, dtype=torch.long)

        i = 0
        for ex in batch:
            if ex is not None:
                seg = ex["seg"]
                feats[i, :len(seg)] = torch.from_numpy(seg)
                feat_lens[i] = len(seg)
                i += 1

        return {
            "view1": feats, "view1_lens": feat_lens,
            "ids": ids, "starts": starts, "ends": ends
        }

    def loader(self, batch_size=1, shuffle=False):
        return tud.DataLoader(self,
                              batch_size=batch_size,
                              collate_fn=self.collate_fn,
                              shuffle=shuffle,
                              num_workers=0)


class FineTuneBatchSampler:
    def __init__(self, pos_examples, neg_examples, batch_size=1, balanced=None, shuffle=False):

        # balanced is the number of neg example given one positive example. For balanced set, use 1

        self.pos_examples = pos_examples
        self.neg_examples = neg_examples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.balanced = balanced

        self.n_pos, self.n_neg = len(self.pos_examples), len(self.neg_examples)

        self.init_iter()

    def init_iter(self):
        self.iter = 0
        self.batches = []

        neg_pairs = self.neg_examples

        if self.balanced is not None:
            neg_pairs = [self.neg_examples[i] for i in np.random.randint(self.n_neg,
                                                                         size=int(self.n_pos * self.balanced))]

        all_examples = [(q, s, 0) for q, s in neg_pairs] + [(q, s, 1) for q, s in self.pos_examples]

        if self.shuffle:
            np.random.shuffle(all_examples)

        batch_size = 0
        batch = []

        for ex in all_examples:
            if batch_size + 1 <= self.batch_size:
                batch.append(ex)
                batch_size += 1
            else:
                if len(batch) > 0:
                    self.batches.append(batch)
                batch = [ex]
                batch_size = 1

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        while self.iter < len(self):
            batch = self.batches[self.iter]
            self.iter += 1
            yield batch
        self.init_iter()

    def state_dict(self, itr):
        return {
            "iter": self.iter - (itr._send_idx - itr._rcvd_idx),
            "batches": np.array(self.batches)
        }

    def load_state_dict(self, state_dict):
        self.iter = state_dict["iter"]
        self.batches = state_dict["batches"].tolist()


class FineTuneDataset(tud.Dataset):

    # the dataset only yield examples from "match" dictionary, which can be used to identify train and dev set

    def __init__(self, query_feats_fn, query_align_fn, search_feats_fn, search_align_fn,
                 match, stack_frames=False, batch_size=1, balanced=False, shuffle=False):

        super().__init__()
        self.stack_frames = stack_frames

        log.info(f"Using {query_feats_fn}; {search_feats_fn}; stacked={stack_frames}")
        self.query_feats = h5py.File(query_feats_fn, "r")
        self.search_feats = h5py.File(search_feats_fn, "r")

        log.info(f"Using {query_align_fn}; {search_align_fn}")
        self.query_align = h5py.File(query_align_fn, "r")
        self.search_align = h5py.File(search_align_fn, "r")

        self.query_examples = {}

        for ex in self.query_align:
            starts = self.query_align[ex]["starts"][()]
            ends = self.query_align[ex]["ends"][()]

            for start, end in zip(starts, ends):

                if stack_frames:
                    start = start // 2
                    end = end // 2

                self.query_examples[ex] = (start, end)

        self.pos_pairs = []
        self.neg_pairs = []

        for (q, s), hit in match.items():

            if hit == 1:
                self.pos_pairs.append((q, s))
            else:
                self.neg_pairs.append((q, s))

        # get feature dimension
        seg = self.query_feats[list(self.query_examples.keys())[0]][()]
        # seg = utils.speech_utils.add_deltas(seg)
        if self.stack_frames:
            seg = utils.speech_utils.stack(seg)
        self.feat_dim = seg.shape[1]

        self.sampler = FineTuneBatchSampler(self.pos_pairs, self.neg_pairs, batch_size=batch_size,
                                            balanced=balanced, shuffle=shuffle)

    def __getitem__(self, pair):

        q_ex, s_ex, hit = pair
        q_start, q_end = self.query_examples[q_ex]

        q_seg = self.query_feats[q_ex][()]

        s_seg = self.search_feats[s_ex][()]

        if self.stack_frames:
            q_seg = utils.speech_utils.stack(q_seg)
            s_seg = utils.speech_utils.stack(s_seg)

        return {"q_seg": q_seg, "q_start": q_start, "q_end": q_end,
                "s_seg": s_seg, "hit": hit}

    def collate_fn(self, batch):
        batch_size = len(batch)
        max_q_seg_dur = 0
        max_s_seg_dur = 0

        q_starts = []
        q_ends = []
        hits = []

        for ex in batch:
            max_q_seg_dur = max(max_q_seg_dur, len(ex["q_seg"]))
            max_s_seg_dur = max(max_s_seg_dur, len(ex["s_seg"]))
            q_starts.append(ex["q_start"])
            q_ends.append(ex["q_end"])
            hits.append(ex["hit"])

        hits = torch.tensor(hits, dtype=torch.long)

        query_feats = torch.zeros(batch_size, max_q_seg_dur, self.feat_dim)
        query_lens = torch.zeros(batch_size, dtype=torch.long)
        search_feats = torch.zeros(batch_size, max_s_seg_dur, self.feat_dim)
        search_lens = torch.zeros(batch_size, dtype=torch.long)

        for ind, ex in enumerate(batch):
            query_feats[ind, :len(ex["q_seg"])] = torch.from_numpy(ex["q_seg"])
            query_lens[ind] = len(ex["q_seg"])
            search_feats[ind, :len(ex["s_seg"])] = torch.from_numpy(ex["s_seg"])
            search_lens[ind] = len(ex["s_seg"])

        return {"query_feats": query_feats, "query_lens": query_lens,
                "q_starts": q_starts, "q_ends": q_ends,
                "search_feats": search_feats, "search_lens": search_lens,
                "hits": hits}

    def loader(self):
        return tud.DataLoader(self,
                              batch_sampler=self.sampler,
                              collate_fn=self.collate_fn,
                              num_workers=0)


class CNNDataset(tud.Dataset):

    # the dataset only yield examples from "match" dictionary, which can be used to identify train and dev set

    def __init__(self, query_hidden_fn, search_hidden_fn,
                 match, batch_size=1, balanced=None, shuffle=False,
                 maxWidth=100, maxLength=800):

        super().__init__()

        self.maxWidth = maxWidth
        self.maxLength = maxLength

        self.query_dict = np.load(query_hidden_fn, allow_pickle=True).item()
        self.search_dict = np.load(search_hidden_fn, allow_pickle=True).item()

        qks = self.query_dict.keys()
        sks = self.search_dict.keys()

        self.pos_pairs = []
        self.neg_pairs = []

        for (q, s), hit in match.items():

            if hit == 1:
                self.pos_pairs.append((q, s))
            else:
                self.neg_pairs.append((q, s))

        self.sampler = FineTuneBatchSampler(self.pos_pairs, self.neg_pairs, batch_size=batch_size,
                                            balanced=balanced, shuffle=shuffle)

    def __getitem__(self, pair):

        q_ex, s_ex, hit = pair

        return {"q_seg": self.query_dict[q_ex],
                "s_seg": self.search_dict[s_ex],
                "hit": hit}

    def compression_index(self, length, max_length):
        # no of elements to be deleted
        n_del = length - max_length
        if n_del > 0:
            # index of the elements to be deleted
            ind_del = (length / n_del) * np.array(range(n_del))
            # index of the elements to choose for compression
            ind_keep = np.delete(np.array(range(length)), ind_del, axis=0)
        else:
            ind_keep = np.array(range(length))
        return ind_keep

    def collate_fn(self, batch):

        batch_size = len(batch)
        out = torch.zeros(batch_size, 1, self.maxWidth, self.maxLength).fill_(-1)
        hits = []

        for i, ex in enumerate(batch):
            query_hidden = torch.from_numpy(ex["q_seg"])
            search_hidden = torch.from_numpy(ex["s_seg"])

            width = query_hidden.shape[0]
            length = search_hidden.shape[0]
            ind_width = torch.LongTensor(self.compression_index(width, self.maxWidth))
            ind_length = torch.LongTensor(self.compression_index(length, self.maxLength))

            query_hidden = query_hidden.index_select(0, ind_width)[:, None, :]
            search_hidden = search_hidden.index_select(0, ind_length)[None, :, :]

            sim = torch.nn.functional.cosine_similarity(query_hidden, search_hidden, dim=-1)

            sim = -1 + 2* (sim - sim.min())/(sim.max() - sim.min())

            out[i, 0].narrow(0, (self.maxWidth - sim.size(0)) // 2, sim.size(0)).narrow(1, (
                        self.maxLength - sim.size(1)) // 2, sim.size(1)).copy_(sim)

            hits.append(ex["hit"])

        hits = torch.tensor(hits, dtype=torch.long)

        return out, hits

    def loader(self):
        return tud.DataLoader(self,
                              batch_sampler=self.sampler,
                              collate_fn=self.collate_fn,
                              num_workers=0)


class CNNEvalset(tud.Dataset):

    # the dataset only yield examples from "match" dictionary, which can be used to identify train and dev set

    def __init__(self, query_hidden_fn, search_hidden_fn,
                batch_size=1,  shuffle=False,
                 maxWidth=100, maxLength=800):

        super().__init__()

        self.maxWidth = maxWidth
        self.maxLength = maxLength
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.query_dict = np.load(query_hidden_fn, allow_pickle=True).item()
        self.search_dict = np.load(search_hidden_fn, allow_pickle=True).item()

        qks = self.query_dict.keys()
        sks = self.search_dict.keys()

        self.pairs = []

        for q in qks:
            for s in sks:
                self.pairs.append((q, s))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        q_ex, s_ex = self.pairs[idx]

        return {"q_seg": self.query_dict[q_ex],
                "s_seg": self.search_dict[s_ex],
                "pair": (q_ex, s_ex)}

    def compression_index(self, length, max_length):
        # no of elements to be deleted
        n_del = length - max_length
        if n_del > 0:
            # index of the elements to be deleted
            ind_del = (length / n_del) * np.array(range(n_del))
            # index of the elements to choose for compression
            ind_keep = np.delete(np.array(range(length)), ind_del, axis=0)
        else:
            ind_keep = np.array(range(length))
        return ind_keep

    def collate_fn(self, batch):

        batch_size = len(batch)
        out = torch.zeros(batch_size, 1, self.maxWidth, self.maxLength).fill_(-1)
        pairs = []

        for i, ex in enumerate(batch):
            query_hidden = torch.from_numpy(ex["q_seg"])
            search_hidden = torch.from_numpy(ex["s_seg"])

            width = query_hidden.shape[0]
            length = search_hidden.shape[0]
            ind_width = torch.LongTensor(self.compression_index(width, self.maxWidth))
            ind_length = torch.LongTensor(self.compression_index(length, self.maxLength))

            query_hidden = query_hidden.index_select(0, ind_width)[:, None, :]
            search_hidden = search_hidden.index_select(0, ind_length)[None, :, :]

            sim = torch.nn.functional.cosine_similarity(query_hidden, search_hidden, dim=-1)

            sim = -1 + 2 * (sim - sim.min()) / (sim.max() - sim.min())

            out[i, 0].narrow(0, (self.maxWidth - sim.size(0)) // 2, sim.size(0)).narrow(1, (
                    self.maxLength - sim.size(1)) // 2, sim.size(1)).copy_(sim)

            pairs.append(ex["pair"])

        return out, pairs

    def loader(self):
        return tud.DataLoader(self,
                              batch_size=self.batch_size,
                              shuffle=self.shuffle,
                              collate_fn=self.collate_fn,
                              num_workers=0)



"""
class SearchDataset(tud.Dataset):

    def __init__(self, feats_fn, align_fn, stack_frames=False, min_search_len=10):

        super().__init__()
        self.stack_frames = stack_frames

        log.info(f"Using {feats_fn}; stacked={stack_frames}")
        self.feats = h5py.File(feats_fn, "r")

        log.info(f"Using {align_fn}")
        self.align = h5py.File(align_fn, "r")

        self.examples = []

        for ex in self.align:
            starts = self.align[ex]["starts"][()]
            ends = self.align[ex]["ends"][()]
            # some end-start < 10

            for start, end in zip(starts, ends):

                if end - start > min_search_len:
                    self.examples.append((ex, start, end))

        # get feature dimension
        seg = self.feats[self.examples[0][0]][()]
        # seg = utils.speech_utils.add_deltas(seg)
        if self.stack_frames:
            seg = utils.speech_utils.stack(seg)
        self.feat_dim = seg.shape[1]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        ex, start, end = self.examples[idx]

        seg = self.feats[ex][()][start:end + 1, :]

        if len(seg) == 0:
            return None

        # seg = utils.speech_utils.add_deltas(seg)

        if self.stack_frames:
            seg = utils.speech_utils.stack(seg)

        return {"seg": seg, "query": ex}

    def collate_fn(self, batch):

        batch_size = 0
        max_seg_dur = 0

        ids = []
        for ex in batch:
            if ex is not None:
                max_seg_dur = max(max_seg_dur, len(ex["seg"]))
                ids.append(ex["query"])
                batch_size += 1

        feats = torch.zeros(batch_size, max_seg_dur, self.feat_dim)
        feat_lens = torch.zeros(batch_size, dtype=torch.long)

        i = 0
        for ex in batch:
            if ex is not None:
                seg = ex["seg"]
                feats[i, :len(seg)] = torch.from_numpy(seg)
                feat_lens[i] = len(seg)
                i += 1

        return {
            "view1": feats, "view1_lens": feat_lens,
            "ids": ids
        }

    def loader(self, batch_size=1, shuffle=False):
        return tud.DataLoader(self,
                              batch_size=batch_size,
                              collate_fn=self.collate_fn,
                              shuffle=shuffle,
                              num_workers=0)
"""
