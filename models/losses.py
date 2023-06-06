import torch.nn.functional as F
import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    
    def __init__(self, reduce_zero_label=False):
        super(DiceLoss, self).__init__()
        print("reduce_zero_label:", reduce_zero_label)
        self.reduce_zero_label = reduce_zero_label

    def forward(self, input, target, reduce=True):
        input = torch.sigmoid(input)
        input = input.reshape(-1)
        target = target.reshape(-1).float()
        mask = ~torch.isnan(target)
        if self.reduce_zero_label:
            target = target - 1  # start from zero
        input = input[mask]
        target = target[mask]
        
        a = torch.sum(input * target)
        b = torch.sum(input * input) + 0.001
        c = torch.sum(target * target) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        if reduce:
            loss = torch.mean(loss)

        return loss


class BinaryCrossEntropyLoss(nn.Module):
    
    def __init__(self, weight, reduce_zero_label=False):
        super(BinaryCrossEntropyLoss, self).__init__()
        print("weight:", weight, "reduce_zero_label:", reduce_zero_label)
        if weight is not None:
            self.weight = torch.tensor(weight).cuda()
        else:
            self.weight = None
        self.reduce_zero_label = reduce_zero_label
            
    def forward(self, outputs, targets):
        # if outputs.size(-1) == 1 and len(outputs.shape) > 1:
        #     outputs = outputs.squeeze(-1)
        outputs = outputs.reshape(-1)
        targets = targets.reshape(-1)
        mask = ~torch.isnan(targets)
        if self.reduce_zero_label:
            targets = targets - 1  # start from zero
        if self.weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                outputs[mask].float(), targets[mask].float(), self.weight[targets[mask].long()], reduction="sum")
        else:
            loss = F.binary_cross_entropy_with_logits(
                outputs[mask].float(), targets[mask].float(), reduction="sum")
        sample_size = torch.sum(mask.type(torch.int64))
        return loss / sample_size


class CrossEntropyLoss(nn.Module):
    
    def __init__(self, weight, reduce_zero_label=False):
        super(CrossEntropyLoss, self).__init__()
        print("weight:", weight, "reduce_zero_label:", reduce_zero_label)
        if weight is not None:
            self.weight = torch.tensor(weight).cuda()
        else:
            self.weight = None
        self.reduce_zero_label = reduce_zero_label
    
    
    def forward(self, outputs, targets):
        mask = ~torch.isnan(targets)
        if self.reduce_zero_label:
            targets = targets - 1  # start from zero
        if self.weight is not None:
            loss = F.cross_entropy(
                outputs[mask].float(), targets[mask].long(), self.weight, reduction="sum")
        else:
            loss = F.cross_entropy(
                outputs[mask].float(), targets[mask].long(), reduction="sum")
        sample_size = torch.sum(mask.type(torch.int64))
        return loss / sample_size
        
