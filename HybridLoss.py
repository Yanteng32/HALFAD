import torch
import torch.nn.functional as F


class HybridLoss(torch.nn.Module):
    def __init__(self, alpha_p=1.0, alpha_r=1.0):
        super(HybridLoss, self).__init__()
        self.alpha_p = alpha_p
        self.alpha_r = alpha_r

    def forward(self, P_p, P_r, P_s, y_true):
        """
        :param P_p: Patch-level probability predictions, shape: [batch_size, num_classes]
        :param P_r: Region-level probability predictions, shape: [batch_size, num_classes]
        :param P_s: Subject-level probability predictions, shape: [batch_size, num_classes]
        :param y_true: True labels, shape: [batch_size]
        """
        # Convert true labels to one-hot encoding
        y_one_hot = F.one_hot(y_true, num_classes=2).float()

        # Patch-level loss
        if P_p is not None:
            loss_p = -torch.mean(torch.sum(y_one_hot * torch.log(P_p + 1e-5), dim=1))
        else:
            loss_p = 0

        # Region-level loss
        if P_r is not None:
            loss_r = -torch.mean(torch.sum(y_one_hot * torch.log(P_r + 1e-5), dim=1))
        else:
            loss_r = 0

        # Subject-level loss
        if P_s is not None:
            loss_s = -torch.mean(torch.sum(y_one_hot * torch.log(P_s + 1e-5), dim=1))
        else:
            loss_s = 0

        # Hybrid loss
        total_loss = self.alpha_p * loss_p + self.alpha_r * loss_r + loss_s

        return total_loss


