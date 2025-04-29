import torch
import segmentation_models_pytorch as smp

def build_model(encoder='resnet34', encoder_weights='imagenet', num_classes=1, activation='sigmoid'):
    print(f"Создание модели U-Net (PyTorch) с энкодером {encoder}...")
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=activation,
    )
    print("Модель создана.")
    return model