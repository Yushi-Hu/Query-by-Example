import logging as log
import os
import json
import argparse
import random
import numpy as np
import torch
import torch.utils.tensorboard as tensorboard

import net
import loss
import metric
import data

import optim
import sched
import utils.saver

from statistics import mean


class Trainer(utils.saver.TrainerSaver):
    savable = ["segnet", "optim", "sched", "data"]

    def __init__(self, config_file, config):
        super().__init__(cache=False)

        self.train_language_list = config.train_language_list
        self.n_train_lang = len(config.train_language_list)

        self.main_dev_list = config.main_dev_language_list
        self.n_main_dev_lang = len(config.main_dev_language_list)

        self.add_dev_list = config.add_dev_language_list
        self.n_add_dev_lang = len(config.add_dev_language_list)

        # self.subwords_to_ids = data.combine_subwords_to_ids(config.all_vocab, config.subwords)
        with open(config.subwords_to_ids, 'r') as f:
            self.subwords_to_ids = json.load(f)

        self.data = data.MultilangDataset(feats_fns=config.train_feats,
                                          align_fns=config.train_align,
                                          vocab_fns=config.train_vocab,
                                          subwords=config.subwords,
                                          subwords_to_ids=self.subwords_to_ids,
                                          min_word_dur=config.train_min_word_dur,
                                          min_seg_dur=config.train_min_seg_dur,
                                          stack_frames=config.stack_frames,
                                          batch_size=config.train_batch_size,
                                          shuffle=config.shuffle,
                                          variable=config.variable_batch,
                                          cache=self.cache,
                                          spec_aug=True)

        # statistics
        train_subwords = set(data.combine_subwords_to_ids(config.train_vocab, config.subwords).keys())
        log.info(f"Using {len(train_subwords)} subwords in training")

        # dev sets for all training languages
        self.dev_datasets = []

        for i in range(self.n_main_dev_lang + self.n_add_dev_lang):
            data_dev = data.MultilangDataset(feats_fns=[config.dev_feats[i]],
                                             align_fns=[config.dev_align[i]],
                                             vocab_fns=[config.dev_vocab[i]],
                                             subwords=config.subwords,
                                             subwords_to_ids=self.subwords_to_ids,
                                             min_word_dur=config.dev_min_word_dur,
                                             min_seg_dur=config.dev_min_seg_dur,
                                             stack_frames=config.stack_frames,
                                             batch_size=config.dev_batch_size,
                                             variable=config.variable_batch,
                                             cache=self.cache,
                                             shuffle=False,
                                             spec_aug=False)

            self.dev_datasets.append(data_dev)

            # statistics
            if i < self.n_main_dev_lang:
                this_lang = self.main_dev_list[i]
            else:
                this_lang = self.add_dev_list[i - self.n_main_dev_lang]

            this_subwords = set(data.combine_subwords_to_ids([config.dev_vocab[i]], config.subwords))
            log.info(f"language {this_lang} has {len(this_subwords)} subwords, "
                     f"intersect {len(train_subwords.intersection(this_subwords))} subwords")

        loss_fun = loss.Obj02(margin=config.loss_margin)

        self.awe = net.MultiViewRNN(config=config,
                                    feat_dim=self.data.feat_dim,
                                    num_subwords=self.data.num_subwords,
                                    use_gpu=True)

        self.segnet = net.MultiviewSpanRNN(config=config, awe=self.awe, loss_fun=loss_fun, use_gpu=True)

        self.optim = optim.Adam(params=self.segnet.parameters(), lr=config.adam_lr)

        self.sched = sched.RevertOnPlateau(network=self.segnet,
                                           optimizer=self.optim,
                                           mode=config.mode,
                                           factor=config.factor,
                                           patience=config.patience,
                                           min_lr=config.min_lr)

        expt_dir = os.path.dirname(config_file)
        save_dir = os.path.join(expt_dir, "save")

        self.save_dir = save_dir
        self.set_savepaths(save_dir=save_dir)

        self.config_file = config_file
        self.config = config

        self.pred_loss = loss.BinaryFocalLoss(reduction="mean")

    @property
    def global_step(self):
        return self.config.global_step

    @property
    def fine_tune(self):
        return self.config.fine_tune == "true"


class SpanId:

    def __init__(self):
        self.span_dict = {}
        self.new_id = 0

    def get_index(self, span):
        span = tuple(span)
        if span in self.span_dict:
            return self.span_dict[span]
        else:
            self.span_dict[span] = self.new_id
            self.new_id += 1
            return self.new_id - 1


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s: %(message)s")
    # torch.backends.cudnn.enabled = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration filename")
    args = parser.parse_args()

    config_file = args.config
    with open(config_file, "r") as f:
        config = argparse.Namespace(**json.load(f))

    random.seed(config.global_step)
    np.random.seed(config.global_step)
    torch.manual_seed(config.global_step)

    trainer = Trainer(config_file, config)

    # load AWE
    trainer.awe.set_savepath(save_dir=trainer.save_dir, name='net')
    trainer.awe.load(tag='ft')

    if trainer.global_step > 0:
        trainer.load()

    writer = tensorboard.SummaryWriter(
        log_dir=os.path.join(trainer.save_dir, "tensorboard"))

    eval_interval = trainer.config.eval_interval or len(trainer.data)

    # initialize number of negative examples
    this_k = config.loss_maxk + config.loss_k_decrease

    while not trainer.optim.converged:

        # reduce number of negative sample per epoch
        this_k -= config.loss_k_decrease
        this_k = max(this_k, config.loss_mink)

        for iter_, batch in enumerate(trainer.data, trainer.data.iter):

            trainer.segnet.train()

            # if a batch is not legal, skip it
            if batch is None:
                continue

            trainer.optim.zero_grad()

            acoustic_out, written_out = trainer.segnet.forward(batch, return_utt_hidden=True)

            span_contrast_loss = trainer.segnet.loss_fun(acoustic_out["span_emb"], written_out["span_emb"],
                                                         batch["span_inv"], this_k)

            legal_word_inds = batch["legal_word_inds"]
            word_contrast_loss = trainer.segnet.loss_fun(acoustic_out["word_emb"][legal_word_inds],
                                                         written_out["word_emb"],
                                                         batch["word_inv"][legal_word_inds], this_k)

            n_unique_span = len(batch['span_uids'])
            n_span = len(batch['span_inv'])

            written_query_scores = n_unique_span * trainer.segnet.pred_score(acoustic_out["utt_hidden"],
                                                                                        written_out["span_emb"])
            acoustic_query_scores = n_span * trainer.segnet.pred_score(acoustic_out["utt_hidden"],
                                                                                         acoustic_out["span_emb"])

            device = acoustic_query_scores.device

            written_query_loss = trainer.pred_loss(written_query_scores.view(-1, 1),
                                                   batch["utt_uspan_gold"].view(-1, 1).to(device))

            acoustic_query_loss = trainer.pred_loss(acoustic_query_scores.view(-1, 1),
                                                    batch["utt_allspan_gold"].view(-1, 1).to(device))

            grad_norm = trainer.segnet.backward(span_contrast_loss + word_contrast_loss +
                                                written_query_loss + acoustic_query_loss)
            trainer.optim.step()

            n_utt = batch["view1"].shape[0]

            log.info(f"batch {iter_}) "
                     f"global_step={trainer.global_step}, "
                     f"pc_loss={span_contrast_loss.item():.2f}, "
                     f"wc_loss={word_contrast_loss.item():.2f}, "
                     f"v1q_loss={acoustic_query_loss.item():.2f}, "
                     f"v2q_loss={written_query_loss.item():.2f}, "
                     f"grad_norm={grad_norm:.2f}, "
                     f"utts={len(batch['view1_lens'])}, "
                     f"phrs={n_span}, "
                     f"ws={len(batch['word_inv'])}, "
                     f"ups={n_unique_span}, "
                     f"uws={len(batch['word_uids'])}")

            trainer.config.global_step += 1

            if trainer.global_step % eval_interval == 0:
                trainer.segnet.eval()

                span2id = SpanId()

                # average ap of languages
                main_acoustic_aps = np.zeros(trainer.n_main_dev_lang)
                main_crossview_aps = np.zeros(trainer.n_main_dev_lang)

                for i in range(trainer.n_main_dev_lang + trainer.n_add_dev_lang):

                    span_embs1, span_ids1, span_embs2, span_ids2  = [], [], [], []
                    word_embs1, word_ids1, word_embs2, word_ids2 = [], [], [], []

                    span_set = set()

                    total_batch = 0
                    total_ap = 0

                    with torch.no_grad():

                        for batch in trainer.dev_datasets[i].loader:

                            if batch is None:
                                continue

                            spans = batch.pop("span_word_ids")

                            acoustic_out, written_out = trainer.segnet.forward(batch,
                                                                               return_utt_hidden=True, numpy=True)

                            written_query_scores = trainer.segnet.pred_score(acoustic_out["utt_hidden"],
                                                                             written_out["span_emb"])
                            acoustic_query_scores = trainer.segnet.pred_score(acoustic_out["utt_hidden"],
                                                                              acoustic_out["span_emb"])

                            acoustic_query_scores = acoustic_query_scores.detach().cpu().numpy()
                            written_query_scores = written_query_scores.detach().cpu().numpy()

                            utt_uspan_gold = batch["utt_uspan_gold"].numpy()
                            utt_allspan_gold = batch["utt_allspan_gold"].numpy()

                            total_ap += metric.class_ap(acoustic_query_scores, utt_allspan_gold)
                            total_ap += metric.class_ap(written_query_scores, utt_uspan_gold)
                            total_batch += 2


                            span_ids = []
                            for span in spans:
                                span_ids.append(span2id.get_index(span))

                            span_ids = np.array(span_ids, dtype=int)
                            span_ids1.append(span_ids[batch["span_inv"].numpy()])
                            span_ids2.append(span_ids)
                            span_embs1.append(acoustic_out["span_emb_np"])
                            span_embs2.append(written_out["span_emb_np"])

                            legal_word_inds = batch["legal_word_inds"]
                            word_ids = batch["word_uids"]
                            word_ids1.append(word_ids[(batch["word_inv"].numpy())[legal_word_inds]])
                            word_ids2.append(word_ids)
                            word_embs1.append(acoustic_out["word_emb_np"][legal_word_inds])
                            word_embs2.append(written_out["word_emb_np"])

                    span_ids1 = np.hstack(span_ids1)
                    span_ids2, ind = np.unique(np.hstack(span_ids2), return_index=True)
                    span_embs1 = np.vstack(span_embs1)
                    span_embs2 = np.vstack(span_embs2)[ind]

                    word_ids1 = np.hstack(word_ids1)
                    word_ids2, ind = np.unique(np.hstack(word_ids2), return_index=True)
                    word_embs1 = np.vstack(word_embs1)
                    word_embs2 = np.vstack(word_embs2)[ind]


                    span_acoustic_ap = metric.acoustic_ap(span_embs1, span_ids1)
                    span_crossview_ap = metric.crossview_ap(span_embs1, span_ids1, span_embs2, span_ids2)

                    word_acoustic_ap = metric.acoustic_ap(word_embs1, word_ids1)
                    word_crossview_ap = metric.crossview_ap(word_embs1, word_ids1, word_embs2, word_ids2)


                    if i < trainer.n_main_dev_lang:
                        this_lang = trainer.main_dev_list[i]
                        main_acoustic_aps[i] = span_acoustic_ap
                        main_crossview_aps[i] = span_crossview_ap

                        log.info(f"epoch {trainer.global_step / len(trainer.data):.2f}) "
                                 f"global_step={trainer.global_step}, "
                                 f"language={this_lang},"
                                 f"span_acoustic_ap={span_acoustic_ap:.2f}, "
                                 f"span_crossview_ap={span_crossview_ap:.2f} "
                                 f"word_acoustic_ap={word_acoustic_ap:.2f}, "
                                 f"word_crossview_ap={word_crossview_ap:.2f} "
                                 f"avg_ap={total_ap / total_batch}")

                        """
                        # save embeddings. Don't save for now
                        words1 = map(trainer.data_dev.ids_to_words.get, ids1)
                        writer.add_embedding(embs1, metadata=list(words1), tag="view1_embs")
                        words2 = map(trainer.data_dev.ids_to_words.get, ids2)
                        writer.add_embedding(embs2, metadata=list(words2), tag="view2_embs")
                        """

                        if i == (trainer.n_main_dev_lang - 1):
                            avg_cross_ap = np.mean(main_crossview_aps)
                            avg_acoustic_ap = np.mean(main_acoustic_aps)
                            log.info(f"epoch {trainer.global_step / len(trainer.data):.2f}) "
                                     f"global_step={trainer.global_step}, "
                                     f"avg_span_acoustic_ap={avg_acoustic_ap:.2f}, "
                                     f"avg_span_crossview_ap={avg_cross_ap:.2f} ")

                            best_so_far = trainer.sched.step(avg_cross_ap, trainer.global_step)

                            if best_so_far:
                                log.info("crossview_ap best")
                                trainer.save(best=True)

                    # additional evaluated language
                    else:
                        this_lang = trainer.add_dev_list[i - trainer.n_main_dev_lang]
                        log.info(f"epoch {trainer.global_step / len(trainer.data):.2f}) "
                                 f"global_step={trainer.global_step}, "
                                 f"language={this_lang},"
                                 f"span_acoustic_ap={span_acoustic_ap:.2f}, "
                                 f"span_crossview_ap={span_crossview_ap:.2f} "
                                 f"word_acoustic_ap={word_acoustic_ap:.2f}, "
                                 f"word_crossview_ap={word_crossview_ap:.2f} "
                                 f"avg_ap={total_ap / total_batch}")
