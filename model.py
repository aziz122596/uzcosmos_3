import torch
import segmentation_models_pytorch as smp

def build_model(encoder='resnet34', encoder_weights='imagenet', num_classes=1, activation='sigmoid'):
    """
    Создает модель сегментации U-Net с использованием segmentation-models-pytorch.
    """
    print(f"Создание модели U-Net (PyTorch) с энкодером {encoder}...")
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=activation, # sigmoid для бинарной классификации с Dice/Jaccard/BCE loss
                              
    )
    print("Модель создана.")
    return model