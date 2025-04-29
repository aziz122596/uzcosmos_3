from segmentation_models import Unet 

def build_unet_model(backbone='resnet34', input_shape=(256, 256, 3), classes=1, activation='sigmoid', encoder_weights='imagenet', encoder_freeze=False):
    """
    Строит модель U-Net с заданным бэкбоном.
    Args:
        backbone (str): Имя энкодера (бэкбона).
        input_shape (tuple): Размер входного изображения (H, W, C).
        classes (int): Количество классов сегментации.
        activation (str): Функция активации последнего слоя.
        encoder_weights (str): Веса для энкодера ('imagenet' или None).
        encoder_freeze (bool): Заморозить ли веса энкодера.
    Returns:
        keras.Model: Модель Keras.
    """
    print(f"Создание модели U-Net с бэкбоном {backbone}...")
    model = Unet(
        backbone_name=backbone,
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        encoder_weights=encoder_weights,
        encoder_freeze=encoder_freeze
    )
    print("Модель создана.")
    return model