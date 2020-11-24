import logging as log
import torch
import torch.nn.functional as F

import layers
import utils.saver


class MultiViewRNN(utils.saver.NetSaver):

    def __init__(self, config, feat_dim, num_subwords,
                 loss_fun=None, use_gpu=False):
        super(MultiViewRNN, self).__init__()

        self["view1"] = layers.rnn.RNN_default(cell=config.view1_cell,
                                               inputs=feat_dim,
                                               hidden=config.view1_hidden,
                                               bidir=config.view1_bidir,
                                               dropout=config.view1_dropout,
                                               layers=config.view1_layers)

        self["view2"] = layers.rnn.RNN_default(cell=config.view2_cell,
                                               num_embeddings=num_subwords,
                                               inputs=config.view2_inputs,
                                               hidden=config.view2_hidden,
                                               bidir=config.view2_bidir,
                                               dropout=config.view2_dropout,
                                               layers=config.view2_layers)

        log.info(f"view1: feat_dim={feat_dim}")
        log.info(f"view2: num_subwords={num_subwords}")

        if config.projection is not None:
            self["proj"] = layers.linear.Linear(in_features=self["view1"].d_out,
                                                out_features=config.projection)

        if loss_fun is not None:
            self.loss_fun = loss_fun

        if use_gpu and torch.cuda.is_available():
            self.cuda()

        log.info(f"On {'GPU' if torch.cuda.is_available() else 'CPU'}")

    def forward_view(self, view, batch, numpy=False):
        if view == "view1":
            if "starts" not in batch:
                _, _, emb = self.net["view1"](batch["view1"], batch["view1_lens"])
            else:
                emb = []
                out, lens, _ = self.net["view1"](batch["view1"], batch["view1_lens"])
                batch_size, seq_len, _ = out.shape
                out = out.view(batch_size, seq_len, 2, -1)
                for i in range(len(out)):
                    for j in range(len(batch["starts"][i])):
                        start = batch["starts"][i][j]
                        end = batch["ends"][i][j]
                        # emb.append(out[i, start:end + 1].mean(0))
                        emb.append(torch.cat((out[i, end - 1, 0], out[i, start, 1]), -1))
                emb = torch.stack(emb)
        if view == "view2":
            _, _, emb = self.net["view2"](batch["view2"], batch["view2_lens"])

        if "proj" in self.net:
            emb = self.net["proj"](emb)

        if numpy:
            return emb.detach().cpu().numpy()

        return emb

    def forward(self, batch, numpy=False):
        view1_out = self.forward_view("view1", batch, numpy=numpy)
        view2_out = self.forward_view("view2", batch, numpy=numpy)
        return view1_out, view2_out

    def backward(self, loss):
        loss.backward()
        return torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)


class MultiviewSpanRNN(utils.saver.NetSaver):

    def __init__(self, config, awe, loss_fun=None, use_gpu=False):
        super().__init__()

        self.span_input_dim = config.view1_hidden

        if config.view1_bidir:
            self.span_input_dim *= 2

        # acoustic word embedding model
        self.awe = awe

        self["span_view1"] = layers.rnn.RNN_default(cell=config.view1_cell,
                                                    inputs=self.span_input_dim,
                                                    hidden=config.view1_hidden,
                                                    bidir=config.view1_bidir,
                                                    dropout=config.view1_dropout,
                                                    layers=config.span_view1_layers)

        self["span_view2"] = layers.rnn.RNN_default(cell=config.view2_cell,
                                                    inputs=self.span_input_dim,
                                                    hidden=config.view2_hidden,
                                                    bidir=config.view2_bidir,
                                                    dropout=config.view2_dropout,
                                                    layers=config.span_view2_layers)

        self["pred_score"] = layers.linear.Linear(1, 1)

        if loss_fun is not None:
            self.loss_fun = loss_fun

        if use_gpu and torch.cuda.is_available():
            self.cuda()

        log.info(f"On {'GPU' if torch.cuda.is_available() else 'CPU'}")

    def forward_view1(self, batch, numpy=False, return_utt_hidden=False):

        # acoustic view
        # pre-trained awe hidden states
        awe_out, lens, _ = self.awe.net["view1"](batch["view1"], batch["view1_lens"])
        awe_out = awe_out.detach()

        # segment encoder hidden states
        out, lens, _ = self.net["span_view1"](awe_out, lens)

        batch_size, seq_len, _ = out.shape
        out = out.view(batch_size, seq_len, 2, -1)

        # get segment embeddings
        span_emb = []
        for i in range(batch_size):
            for j in range(len(batch["span_starts"][i])):
                start = batch["span_starts"][i][j]
                end = batch["span_ends"][i][j]
                span_emb.append(torch.cat((out[i, end - 1, 0], out[i, start, 1]), -1))
        span_emb = torch.stack(span_emb)

        word_emb = []
        for i in range(batch_size):
            for j in range(len(batch["word_starts"][i])):
                start = batch["word_starts"][i][j]
                end = batch["word_ends"][i][j]
                word_emb.append(torch.cat((out[i, end - 1, 0], out[i, start, 1]), -1))
        word_emb = torch.stack(word_emb)

        if not return_utt_hidden:
            if not numpy:
                return {"span_emb": span_emb, "word_emb": word_emb}
            else:
                span_emb_np = span_emb.detach().cpu().numpy()
                word_emb_np = word_emb.detach().cpu().numpy()
                return {"span_emb": span_emb, "word_emb": word_emb,
                        "span_emb_np": span_emb_np, "word_emb_np": word_emb_np}

        else:
            utt_hidden = []
            for i in range(batch_size):
                # shape: seq_len * 2 * D. for dim=1, 0 is forward, 1 is backward
                utt_hidden.append(out[i, :lens[i]])

            if not numpy:
                return {"span_emb": span_emb, "word_emb": word_emb, "utt_hidden": utt_hidden}

            else:
                # utt_hidden_np = [s.detach().cpu().numpy() for s in utt_hidden]  # usually do not need hidden numpy
                span_emb_np = span_emb.detach().cpu().numpy()
                word_emb_np = word_emb.detach().cpu().numpy()
                return {"span_emb": span_emb, "word_emb": word_emb, "utt_hidden": utt_hidden,
                        "span_emb_np": span_emb_np, "word_emb_np": word_emb_np}

    def forward_view2(self, batch, numpy=False):

        # written word view
        _, _, agwe = self.awe.net["view2"](batch["word_view2"], batch["word_view2_lens"])
        agwe = agwe.detach()

        # written span emb
        durs = batch["span_view2_lens"]
        span_inv_view2 = batch["span_word_invs_view2"]
        span_view2_input = torch.zeros(len(durs), max(durs), self.span_input_dim)
        for i, span in enumerate(span_inv_view2):
            for j in range(durs[i]):
                span_view2_input[i, j, :] = agwe[span[j]]

        _, _, span_emb = self.net["span_view2"](span_view2_input, durs)

        # written word emb (dummy layer above AGWE)
        n_word = len(agwe)
        word_view2_input = agwe.reshape(n_word, 1, self.span_input_dim)
        _, _, word_emb = self.net["span_view2"](word_view2_input, torch.ones(n_word).type(torch.long))

        if not numpy:
            return {"span_emb": span_emb, "word_emb": word_emb}

        else:
            word_emb_np = word_emb.detach().cpu().numpy()
            span_emb_np = span_emb.detach().cpu().numpy()

            return {"span_emb": span_emb, "word_emb": word_emb,
                    "span_emb_np": span_emb_np, "word_emb_np": word_emb_np}

    def forward(self, batch, numpy=False, return_utt_hidden=False):
        view1_out = self.forward_view1(batch, numpy=numpy, return_utt_hidden=return_utt_hidden)
        view2_out = self.forward_view2(batch, numpy=numpy)
        return view1_out, view2_out

    def pred_score(self, utt_hidden, span_emb):
        n_utt = len(utt_hidden)
        n_span, d = span_emb.shape
        hd = d // 2

        span_emb = span_emb.view(n_span, 2, hd)

        scores = []
        for utt_id in range(n_utt):
            this_utt = utt_hidden[utt_id]

            fwd_scores = torch.max(F.cosine_similarity(this_utt[:, 0, :].view(-1, 1, hd),
                                                       span_emb[:, 0, :].view(1, n_span, hd), dim=-1), dim=0)[0]

            bkwd_scores = torch.max(F.cosine_similarity(this_utt[:, 1, :].view(-1, 1, hd),
                                                        span_emb[:, 1, :].view(1, n_span, hd), dim=-1), dim=0)[0]

            scores.append(fwd_scores + bkwd_scores)

        scores = torch.stack(scores)

        scores = self.net["pred_score"](scores.view(-1, 1)).view(n_utt, n_span)

        return scores

    def backward(self, loss):
        loss.backward()
        return torch.nn.utils.clip_grad_norm_(self.parameters(), 100.0)


class AcousticSpanRNN(utils.saver.NetSaver):

    def __init__(self, config, feat_dim, loss_fun=None, use_gpu=False):
        super().__init__()

        self.span_input_dim = config.view1_hidden

        if config.view1_bidir:
            self.span_input_dim *= 2

        # acoustic word embedding model
        self["view1"] = layers.rnn.RNN_default(cell=config.view1_cell,
                                               inputs=feat_dim,
                                               hidden=config.view1_hidden,
                                               bidir=config.view1_bidir,
                                               dropout=config.view1_dropout,
                                               layers=config.view1_layers)

        self["span_view1"] = layers.rnn.RNN_default(cell=config.view1_cell,
                                                    inputs=self.span_input_dim,
                                                    hidden=config.view1_hidden,
                                                    bidir=config.view1_bidir,
                                                    dropout=config.view1_dropout,
                                                    layers=config.span_view1_layers)

        self["pred_score"] = layers.linear.Linear(1, 1)

        if loss_fun is not None:
            self.loss_fun = loss_fun

        if use_gpu and torch.cuda.is_available():
            self.cuda()

        log.info(f"On {'GPU' if torch.cuda.is_available() else 'CPU'}")

    def forward(self, batch, numpy=False, extra_dim=False):

        # acoustic view
        # pre-trained awe hidden states
        awe_out, lens, _ = self.net["view1"](batch["view1"], batch["view1_lens"])
        out, lens, _ = self.net["span_view1"](awe_out, lens)

        embs = []
        for i in range(len(out)):
            this_emb = out[i, :lens[i]]

            if extra_dim:
                seq_len = this_emb.shape[0]
                this_emb = this_emb.view(seq_len, 2, -1)

            if numpy:
                this_emb = this_emb.detach().cpu().numpy()

            embs.append(this_emb)

        return embs

    def forward_hidden(self, x, lens, extra_dim=False, numpy=False):
        awe_out, lens, _ = self.net["view1"](x, lens)
        out, lens, _ = self.net["span_view1"](awe_out, lens)

        embs = []
        for i in range(len(out)):
            this_emb = out[i, :lens[i]]

            if extra_dim:
                seq_len = this_emb.shape[0]
                this_emb = this_emb.view(seq_len, 2, -1)

            if numpy:
                this_emb = this_emb.detach().cpu().numpy()

            embs.append(this_emb)

        return embs

    def pred_score(self, utt_hidden, span_emb):
        n_utt = len(utt_hidden)
        n_span, d = span_emb.shape
        hd = d // 2

        span_emb = span_emb.view(n_span, 2, hd)

        scores = []
        for utt_id in range(n_utt):
            this_utt = utt_hidden[utt_id]

            fwd_scores = torch.max(F.cosine_similarity(this_utt[:, 0, :].view(-1, 1, hd),
                                                       span_emb[:, 0, :].view(1, n_span, hd), dim=-1), dim=0)[0]

            bkwd_scores = torch.max(F.cosine_similarity(this_utt[:, 1, :].view(-1, 1, hd),
                                                        span_emb[:, 1, :].view(1, n_span, hd), dim=-1), dim=0)[0]

            scores.append(fwd_scores + bkwd_scores)

        scores = torch.stack(scores)

        scores = self.net["pred_score"](scores.view(-1, 1)).view(n_utt, n_span)

        return scores

    def backward(self, loss):
        loss.backward()
        return torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)


class AcousticWordRNN(utils.saver.NetSaver):

    def __init__(self, config, feat_dim, loss_fun=None, use_gpu=False):
        super().__init__()

        self.span_input_dim = config.view1_hidden

        if config.view1_bidir:
            self.span_input_dim *= 2

        # acoustic word embedding model
        self["view1"] = layers.rnn.RNN_default(cell=config.view1_cell,
                                               inputs=feat_dim,
                                               hidden=config.view1_hidden,
                                               bidir=config.view1_bidir,
                                               dropout=config.view1_dropout,
                                               layers=config.view1_layers)


        if loss_fun is not None:
            self.loss_fun = loss_fun

        if use_gpu and torch.cuda.is_available():
            self.cuda()

        log.info(f"On {'GPU' if torch.cuda.is_available() else 'CPU'}")

    def forward(self, batch, numpy=False, extra_dim=False):

        # acoustic view
        # pre-trained awe hidden states
        out, lens, _ = self.net["view1"](batch["view1"], batch["view1_lens"])

        embs = []
        for i in range(len(out)):
            this_emb = out[i, :lens[i]]

            if extra_dim:
                seq_len = this_emb.shape[0]
                this_emb = this_emb.view(seq_len, 2, -1)

            if numpy:
                this_emb = this_emb.detach().cpu().numpy()

            embs.append(this_emb)

        return embs

    def forward_hidden(self, x, lens, extra_dim=False, numpy=False):
        out, lens, _ = self.net["view1"](x, lens)

        embs = []
        for i in range(len(out)):
            this_emb = out[i, :lens[i]]

            if extra_dim:
                seq_len = this_emb.shape[0]
                this_emb = this_emb.view(seq_len, 2, -1)

            if numpy:
                this_emb = this_emb.detach().cpu().numpy()

            embs.append(this_emb)

        return embs


    def backward(self, loss):
        loss.backward()
        return torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)