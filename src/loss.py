import torch
import torch.nn.functional as F
from torch import nn


# Generalized Cross Entropy Loss
class GCELoss(nn.Module):

    def __init__(self, q=0.7, ignore_index=-100):
        super(GCELoss, self).__init__()
        self.q = q
        self.ignore_index = ignore_index
             
    def forward(self, logits, targets):
        # vanilla cross entropy when q = 0
        if self.q == 0:
            if logits.size(-1) == 1:
                ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                loss = ce_loss(logits.view(-1), targets.float())
            else:
                ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
                loss = ce_loss(logits, targets)
        else:
            # if logits.size(-1) == 1:
            #     pred = torch.sigmoid(logits)
            #     pred = torch.cat((1-pred, pred), dim=-1)
            # else:
            pred = F.softmax(logits, dim=-1)
            ce_loss = nn.NLLLoss(reduction='none')
            loss = -ce_loss(pred,targets)
            loss = (1-loss**self.q) / self.q
        #loss = (loss.view(-1)).sum()
        return loss
