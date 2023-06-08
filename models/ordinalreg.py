

import numpy as np
import torch
from torch import nn
from typing import Optional


def _reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Reduce loss
    Parameters
    ----------
    loss : torch.Tensor, [batch_size, num_classes]
        Batch losses.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    Returns
    -------
    loss : torch.Tensor
        Reduced loss.
    """
    if reduction == 'elementwise_mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f'{reduction} is not a valid reduction')


def cumulative_link_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                         reduction: str = 'elementwise_mean',
                         class_weights: Optional[np.ndarray] = None
                         ) -> torch.Tensor:
    """
    Calculates the negative log likelihood using the logistic cumulative link
    function.
    See "On the consistency of ordinal regression methods", Pedregosa et. al.
    for more details. While this paper is not the first to introduce this, it
    is the only one that I could find that was easily readable outside of
    paywalls.
    Parameters
    ----------
    y_pred : torch.Tensor, [batch_size, num_classes]
        Predicted target class probabilities. float dtype.
    y_true : torch.Tensor, [batch_size, 1]
        True target classes. long dtype.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.
    Returns
    -------
    loss: torch.Tensor
    """
    eps = 1e-15

    y_true = y_true.view(-1, 1)

    likelihoods = torch.clamp(torch.gather(y_pred, 1, y_true), eps, 1 - eps)
    neg_log_likelihood = -torch.log(likelihoods)

    if class_weights is not None:
        # Make sure it's on the same device as neg_log_likelihood
        class_weights = torch.as_tensor(class_weights,
                                        dtype=neg_log_likelihood.dtype,
                                        device=neg_log_likelihood.device)
        neg_log_likelihood *= class_weights[y_true]

    loss = _reduction(neg_log_likelihood, reduction)
    return loss


class CumulativeLinkLoss(nn.Module):
    """
    Module form of cumulative_link_loss() loss function
    Parameters
    ----------
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.
    """

    def __init__(self, reduction: str = 'elementwise_mean',
                 class_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        return cumulative_link_loss(y_pred, y_true,
                                    reduction=self.reduction,
                                    class_weights=self.class_weights)


class LogisticCumulativeLink(nn.Module):
    """
    Converts a single number to the proportional odds of belonging to a class.
    Parameters
    ----------
    num_classes : int
        Number of ordered classes to partition the odds into.
    init_cutpoints : str (default='ordered')
        How to initialize the cutpoints of the model. Valid values are
        - ordered : cutpoints are initialized to halfway between each class.
        - random : cutpoints are initialized with random values.
    """

    def __init__(self, num_classes: int,
                 init_cutpoints: str = 'random') -> None:
        assert num_classes > 2, (
            'Only use this model if you have 3 or more classes'
        )
        super().__init__()
        self.num_classes = num_classes
        self.init_cutpoints = init_cutpoints
        if init_cutpoints == 'ordered':
            num_cutpoints = self.num_classes - 1
            cutpoints = torch.arange(num_cutpoints).float() - num_cutpoints / 2
            self.cutpoints = nn.Parameter(cutpoints)
        elif init_cutpoints == 'random':
            cutpoints = torch.rand(self.num_classes - 1).sort()[0]
            self.cutpoints = nn.Parameter(cutpoints, requires_grad=False)
        else:
            raise ValueError(f'{init_cutpoints} is not a valid init_cutpoints '
                             f'type')

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Equation (11) from
        "On the consistency of ordinal regression methods", Pedregosa et. al.
        """
        sigmoids = torch.sigmoid(self.cutpoints - X)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )
        return link_mat


class AscensionCallback():
    """
    Ensure that each cutpoint is ordered in ascending value.
    e.g.
    .. < cutpoint[i - 1] < cutpoint[i] < cutpoint[i + 1] < ...
    This is done by clipping the cutpoint values at the end of a batch gradient
    update. By no means is this an efficient way to do things, but it works out
    of the box with stochastic gradient descent.
    Parameters
    ----------
    margin : float, (default=0.0)
        The minimum value between any two adjacent cutpoints.
        e.g. enforce that cutpoint[i - 1] + margin < cutpoint[i]
    min_val : float, (default=-1e6)
        Minimum value that the smallest cutpoint may take.
    """
    def __init__(self, margin: float = 0.3, min_val: float = -1.0e6) -> None:
        super().__init__()
        self.margin = margin
        self.min_val = min_val

    def clip(self, module: nn.Module) -> None:
        # NOTE: Only works for LogisticCumulativeLink right now
        # We assume the cutpoints parameters are called `cutpoints`.
        if isinstance(module, LogisticCumulativeLink):
            cutpoints = module.cutpoints.data
            # print(f'Call AscensionCallback {cutpoints}')
            for i in range(cutpoints.shape[0] - 1):
                cutpoints[i].clamp_(self.min_val,
                                    cutpoints[i + 1] - self.margin)

    def __call__(self, net: nn.Module, *args, **kwargs) -> None:
        net.apply(self.clip)


class CutpointsCallback():
    """
    Ensure that each cutpoint is ordered in ascending value.
    e.g.
    .. < cutpoint[i - 1] < cutpoint[i] < cutpoint[i + 1] < ...
    This is done by clipping the cutpoint values at the end of a batch gradient
    update. By no means is this an efficient way to do things, but it works out
    of the box with stochastic gradient descent.
    Parameters
    ----------
    margin : float, (default=0.0)
        The minimum value between any two adjacent cutpoints.
        e.g. enforce that cutpoint[i - 1] + margin < cutpoint[i]
    min_val : float, (default=-1e6)
        Minimum value that the smallest cutpoint may take.
    """
    def __init__(self,) -> None:
        super().__init__()


    def reverse_grad(self, module: nn.Module) -> None:
        # NOTE: Only works for LogisticCumulativeLink right now
        # We assume the cutpoints parameters are called `cutpoints`.
        if isinstance(module, LogisticCumulativeLink):
            cutpoints = module.cutpoints
            cutpoints.requires_grad = not cutpoints.requires_grad

    def __call__(self, net: nn.Module, *args, **kwargs) -> None:
        net.apply(self.reverse_grad)