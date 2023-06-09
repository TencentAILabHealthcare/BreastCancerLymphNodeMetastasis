from .epoch_metric import EpochMetric

def roc_auc_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    # print(y_true.shape, y_pred.shape)
    num_class = y_pred.shape[1]
    if num_class == 2:
        y_pred = y_pred[:, 1]
        y_true = y_true[:, 1]
        return roc_auc_score(y_true, y_pred)
    try:
        ret = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
    except:
        print(y_true, y_pred)
        ret = 0.
    return ret


class ROC_AUC(EpochMetric):
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    accumulating predictions and the ground-truth during an epoch and applying
    `sklearn.metrics.roc_auc_score <http://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_ .
    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
    ROC_AUC expects y to be comprised of 0's and 1's. y_pred must either be probability estimates or confidence
    values. To apply an activation to y_pred, use output_transform as shown below:
    .. code-block:: python
        def activated_output_transform(output):
            y_pred, y = output
            y_pred = torch.sigmoid(y_pred)
            return y_pred, y
        roc_auc = ROC_AUC(activated_output_transform)
    """

    def __init__(self, output_transform=lambda x: x):
        super(ROC_AUC, self).__init__(roc_auc_compute_fn, output_transform=output_transform)