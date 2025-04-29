from segmentation_models.losses import bce_jaccard_loss, bce_dice_loss, dice_loss, jaccard_loss
from segmentation_models.metrics import iou_score, f1_score as sm_dice_score 

LOSSES = {
    "bce_jaccard": bce_jaccard_loss,
    "bce_dice": bce_dice_loss,
    "dice": dice_loss,
    "jaccard": jaccard_loss,
}

METRICS = {
    "iou_score": iou_score,
    "dice_score": sm_dice_score,
}

def get_loss(name):
    if name not in LOSSES:
        raise ValueError(f"Неизвестное имя функции потерь: {name}. Доступные: {list(LOSSES.keys())}")
    return LOSSES[name]

def get_metrics(names):
    metrics_list = []
    for name in names:
        if name not in METRICS:
            raise ValueError(f"Неизвестное имя метрики: {name}. Доступные: {list(METRICS.keys())}")
        metrics_list.append(METRICS[name])
    return metrics_list