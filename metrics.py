import torch

SMOOTH = 1e-6

def iou_score(output, target, threshold=0.5):
    """Вычисляет IoU (Intersection over Union) для PyTorch тензоров."""
    with torch.no_grad():
        # output предполагается вероятностями [0, 1] после sigmoid
        pred = (output > threshold).float()
        target = target.float() # Убедимся, что таргет float
        intersection = (pred * target).sum(dim=(1, 2, 3)) # Суммы по батчу, каналам, H, W
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.mean() # Среднее по батчу

def dice_coeff(output, target, threshold=0.5):
    """Вычисляет Dice Coefficient для PyTorch тензоров."""
    with torch.no_grad():
        pred = (output > threshold).float()
        target = target.float()
        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice = (2. * intersection + SMOOTH) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + SMOOTH)
    return dice.mean()