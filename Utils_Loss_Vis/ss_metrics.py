import torch

def segmentation_metrics(pred, label, num_classes):
    """
    Compute segmentation metrics:
    - Pixel Accuracy
    - Mean Accuracy
    - IoU per class
    - Mean IoU (ignores absent classes)
    - FWIoU (Frequency Weighted IoU)

    Args:
        pred: torch.Tensor (bs, h, w), predicted class indices
        label: torch.Tensor (bs, h, w), ground truth class indices
        num_classes: int, total number of classes
    """

    # Flatten
    pred = pred.view(-1)
    label = label.view(-1)

    # Mask out invalid labels if needed
    mask = (label >= 0) & (label < num_classes)
    pred = pred[mask]
    label = label[mask]

    # Confusion matrix
    hist = torch.bincount(
        num_classes * label + pred,
        minlength=num_classes**2
    ).reshape(num_classes, num_classes).float()

    # Metrics
    acc = torch.diag(hist).sum() / hist.sum()  # Pixel accuracy
    acc_cls = torch.diag(hist) / hist.sum(1).clamp(min=1)  # per-class accuracy
    acc_cls = acc_cls.mean()

    iu = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist)).clamp(min=1)  # IoU per class

    # Adjusted mean IoU: average only over classes present in GT
    valid = hist.sum(1) > 0   # class appears in GT
    mean_iu = iu[valid].mean() if valid.any() else torch.tensor(0.0)

    freq = hist.sum(1) / hist.sum()
    fwiou = (freq[freq > 0] * iu[freq > 0]).sum()

    return {
        "Pixel_Accuracy": acc.item(),
        "Mean_Accuracy": acc_cls.item(),
        "IoU_per_class": iu.tolist(),
        "Mean_IoU": mean_iu.item(),
        "FWIoU": fwiou.item(),
    }
