import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation_pipeline(height, width):
    """Возвращает конвейер аугментаций для обучения."""
    return A.Compose([
        A.Resize(height, width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.25),
        A.IAAEmboss(p=0.25),
        A.Blur(p=0.1, blur_limit=3),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5)
        ], p=0.3),
        A.GaussNoise(p=0.2),
    ])

def get_validation_augmentation_pipeline(height, width):
    """Возвращает конвейер аугментаций для валидации (только ресайз)."""
    return A.Compose([
        A.Resize(height, width),
    ])

def get_preprocessing_pipeline(preprocessing_fn=None):
    """Возвращает конвейер препроцессинга (нормализация + ToTensorV2)."""
    _transform = []
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
    else:
        _transform.append(A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0))
    _transform.append(ToTensorV2())
    return A.Compose(_transform)

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_filenames, # Принимаем полные имена файлов
                 transform=None, preprocessing=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # Фильтруем список, оставляя только img-*.png на всякий случай
        self.image_filenames = [f for f in image_filenames if f.lower().startswith('img-') and f.lower().endswith('.png')]
        self.transform = transform
        self.preprocessing = preprocessing
        print(f"PyTorch Dataset создан: {len(self.image_filenames)} ID найдено (img-*.png).")
        if not self.image_filenames:
            print(f"Предупреждение: Список image_filenames пуст!")

    def __len__(self):
        return len(self.image_filenames)

    def load_image(self, path):
        """Загружает .png изображение."""
        try:
            img = cv2.imread(path)
            if img is None: raise ValueError(f"cv2.imread вернул None для {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить изображение {path}: {e}")
            return None

    def load_mask(self, path):
        """Загружает .png маску."""
        try:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None: raise ValueError(f"cv2.imread вернул None для {path}")
            return mask
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить маску {path}: {e}")
            return None

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, img_filename) # Имя маски совпадает

        image = self.load_image(img_path)
        mask = self.load_mask(mask_path)

        if image is None or mask is None:
            print(f"Пропуск примера из-за ошибки загрузки: {img_filename}")
            dummy_img = torch.zeros((3, 256, 256), dtype=torch.float32)
            dummy_mask = torch.zeros((1, 256, 256), dtype=torch.float32)
            return dummy_img, dummy_mask

        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        sample = {'image': image, 'mask': mask}
        if self.transform:
            try: sample = self.transform(**sample)
            except Exception as e: print(f"Ошибка аугментации для {img_filename}: {e}"); pass

        if self.preprocessing:
            try:
                sample = self.preprocessing(**sample)
                image = sample['image']
                mask = sample['mask']
            except Exception as e:
                print(f"Ошибка препроцессинга для {img_filename}: {e}")
                image = torch.zeros((3, 256, 256), dtype=torch.float32)
                mask = torch.zeros((1, 256, 256), dtype=torch.float32)
                return image, mask

        image = image.float()
        mask = mask.float()

        return image, mask