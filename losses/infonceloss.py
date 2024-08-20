import torch
from torch import nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, reduction = None, temperature = 0.1, n_views = 2, device = 'cuda:0'):
        super().__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.n_views = n_views
        self.device = device

    def forward(self, feature_view_1, feature_view_2):
        assert feature_view_1.shape == feature_view_2.shape
        features = torch.cat([feature_view_1, feature_view_2], dim = 0)

        labels = torch.cat([torch.arange(feature_view_1.shape[0]) for i in range(self.n_views)], dim = 0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.device)

        features = F.normalize(features, dim = 1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype = torch.bool, device = self.device)
        labels = labels[~mask].view(labels.shape[0], -1)

        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim = 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device = self.device)

        logits = logits / self.temperature
        return nn.CrossEntropyLoss(reduction = self.reduction)(logits, labels)