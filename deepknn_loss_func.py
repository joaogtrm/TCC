import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepKNNLoss(nn.Module):
    """
    Implements the loss function described in the "Deep kNN for Medical Image Classification" paper.

    This loss function compares each anchor to its K nearest positive neighbors and
    M nearest negative neighbors, calculating the triplet loss for all K*M combinations.

    Args:
        margin (float, optional): The margin for the triplet loss. Default: 0.2 (as often used in metric learning).
        p (float, optional): The norm degree for pairwise distance. Default: 2.0 (L2 norm).
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.

    Shape:
        - anchors: (N, D) where N is the batch size and D is the embedding dimension.
        - positives: (N, K, D) where K is the number of positive neighbors.
        - negatives: (N, M, D) where M is the number of negative neighbors.
        - Output: A scalar loss value if reduction is 'mean' or 'sum'.
    """
    def __init__(self, margin: float = 0.2, p: float = 2.0, reduction: str = "mean"):
        super(DeepKNNLoss, self).__init__()
        if margin <= 0:
            raise ValueError(f"Expected margin to be greater than 0, got {margin}")
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction type: {reduction}")
            
        self.margin = margin
        self.p = p
        self.reduction = reduction

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        # anchors:   (N, D)
        # positives: (N, K, D)
        # negatives: (N, M, D)

        # To calculate distances, we need to broadcast. Unsqueeze adds a dimension.
        # (N, D) -> (N, 1, D)
        anchors_expanded = anchors.unsqueeze(1)

        # Calculate distance between each anchor and its K positive neighbors
        # (N, 1, D) and (N, K, D) -> results in (N, K, D) after broadcasting
        # The pairwise_distance computes the p-norm over the last dimension (D)
        # Resulting shape: (N, K)
        pos_dist = F.pairwise_distance(anchors_expanded, positives, p=self.p)

        # Calculate distance between each anchor and its M negative neighbors
        # (N, 1, D) and (N, M, D) -> results in (N, M, D) after broadcasting
        # Resulting shape: (N, M)
        neg_dist = F.pairwise_distance(anchors_expanded, negatives, p=self.p)

        # Now, we need to compute the loss for all K*M combinations.
        # We can use broadcasting again.
        # pos_dist: (N, K) -> (N, K, 1)
        # neg_dist: (N, M) -> (N, 1, M)
        pos_dist_expanded = pos_dist.unsqueeze(2)
        neg_dist_expanded = neg_dist.unsqueeze(1)
        
        # The subtraction will broadcast (N, K, 1) and (N, 1, M) to (N, K, M)
        # This creates a matrix of loss values for each anchor.
        loss_matrix = pos_dist_expanded - neg_dist_expanded + self.margin

        # Apply the hinge loss (i.e., max(0, loss))
        # We use clamp(min=0) which is equivalent to ReLU or max(0, x)
        loss_matrix = torch.clamp(loss_matrix, min=0.0)

        # Apply the specified reduction
        if self.reduction == "mean":
            return torch.mean(loss_matrix)
        elif self.reduction == "sum":
            return torch.sum(loss_matrix)
        else: # 'none'
            return loss_matrix

