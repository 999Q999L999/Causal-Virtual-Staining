import torch
from torch import optim
import torch.nn.functional as F

celoss_func = torch.nn.CrossEntropyLoss()

def sim_loss(out0, out1, label):
    features_0 = F.normalize(out0, dim=1)
    features_1 = F.normalize(out1, dim=1)
    similarity = torch.mul(features_0, features_1).sum(dim=1).reshape(-1, 1)
    un_similarity = 1 - similarity
    logit = torch.cat([un_similarity, similarity], dim=1)
    loss = celoss_func(logit, label)
    return loss, logit


# def sim_loss(out0, out1):
#     features_0 = F.normalize(out0, dim=1)
#     features_1 = F.normalize(out1, dim=1)
#     similarity = torch.mul(features_0, features_1).sum(dim=1).reshape(-1, 1)
#     return similarity