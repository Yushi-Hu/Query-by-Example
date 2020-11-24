import torch
import torch.nn as nn
import torch.nn.functional as F


class Obj02:

    def __init__(self, margin):

        self.margin = margin

    def __call__(self, x, y, inv, k):

        n, d = x.shape
        m = y.shape[0]

        k = min(k, m - 1, len(torch.unique(inv)) - 1)

        if k < 1:
            return torch.tensor(0.0, requires_grad=True).cuda()

        # Compute same-pair similarities
        same = F.cosine_similarity(x, y[inv])
        #same = my_cosine_similarity(x, y[inv])

        # Compute all diff-pair similarities
        diff_inv = torch.cat([(inv + i) % m for i in range(1, m)])
        diff = F.cosine_similarity(x.view(n, 1, d), y[diff_inv].view(n, m - 1, d), dim=2).flatten()
        #diff = my_cosine_similarity(x.view(n, 1, d), y[diff_inv].view(n, m - 1, d), dim=2).flatten()

        # Find most offending word per utterance: obj0
        diff_word = diff.view(n, m - 1).topk(k, dim=1)[0]
        most_offending_word = F.relu(self.margin + diff_word - same.unsqueeze(-1)).pow(2).mean(dim=1).sqrt()

        # object 1: most offending word per word
        obj1_diff = F.cosine_similarity(y[inv].view(n, 1, d), y[diff_inv].view(n, m - 1, d), dim=2).flatten()
        #obj1_diff = my_cosine_similarity(y[inv].view(n, 1, d), y[diff_inv].view(n, m - 1, d), dim=2).flatten()
        obj1_word = obj1_diff.view(n, m - 1).topk(k, dim=1)[0]
        obj1_most_offending_word = F.relu(self.margin + obj1_word - same.unsqueeze(-1)).pow(2).mean(dim=1).sqrt()

        # Find most offending utterance per word: obj2
        diff_utt = torch.zeros(m, k, device=diff_word.device)
        for i in range(m):
            diff_utt[i] = diff[diff_inv == i].topk(k)[0]
        most_offending_utt = F.relu(self.margin + diff_utt[inv] - same.unsqueeze(-1)).pow(2).mean(dim=1).sqrt()

        return most_offending_word.sum() + most_offending_utt.sum() + obj1_most_offending_word.sum()


class BinaryFocalLoss(nn.Module):

    def __init__(self, gamma=2., reduction='none'):
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_score, target_tensor):
        input_score = input_score.flatten()
        target_tensor = target_tensor.flatten()

        batch_size = input_score.shape[0]

        input_tensor = torch.zeros(batch_size, 2).to(input_score.device)
        input_tensor[:, 1] = input_score

        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1. - prob) ** self.gamma) * log_prob,
            target_tensor,
            reduction=self.reduction
        )

class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='none'):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, input_score, target_tensor):
        input_score = input_score.flatten()
        target_tensor = target_tensor.flatten()

        batch_size = input_score.shape[0]

        input_tensor = torch.zeros(batch_size, 2).to(input_score.device)
        input_tensor[:, 1] = input_score

        return self.loss(input_tensor, target_tensor)