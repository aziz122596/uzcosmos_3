import os
import numpy as np
from tensorflow import keras # Используем tf.keras
from skimage.io import imread
import albumentations as A
import math
import cv2 

class SegmentationDataGenerator(keras.utils.Sequence):
    """
    Генератор данных для Keras для задач сегментации.
    Читает изображения и маски из папок, используя полные имена файлов.
    """
    def __init__(self, image_dir, mask_dir, image_filenames, batch_size=1, 
                 img_size=(256, 256), n_channels=3, n_classes=1,
                 augmentation=None, preprocessing=None, shuffle_on_epoch_end=True):
        """
        Инициализация генератора.
        Args:
            image_dir (str): Путь к папке с изображениями.
            mask_dir (str): Путь к папке с масками.
            image_filenames (list): Список ПОЛНЫХ имен файлов изображений (e.g., ['img-1.png', ...]).
            batch_size (int): Размер батча.
            img_size (tuple): Размер изображения (height, width).
            n_channels (int): Количество каналов изображения.
            n_classes (int): Количество классов сегментации.
            augmentation (albumentations.Compose, optional): Конвейер аугментаций.
            preprocessing (albumentations.Compose, optional): Конвейер препроцессинга.
            shuffle_on_epoch_end (bool): Перемешивать ли данные в конце эпохи.
        """
        if not os.path.isdir(image_dir): raise ValueError(f"Директория изображений не найдена: {image_dir}")
        if not os.path.isdir(mask_dir): raise ValueError(f"Директория масок не найдена: {mask_dir}")

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = image_filenames 
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.shuffle = shuffle_on_epoch_end
        self.indexes = np.arange(len(self.image_filenames))
        self.on_epoch_end()

        print(f"Генератор создан: {len(self.image_filenames)} сэмплов, размер батча {self.batch_size}.")
        if not self.image_filenames:
             print(f"Предупреждение: Список image_filenames пуст!")

    def __len__(self):
        """Возвращает количество батчей за эпоху."""
        return math.ceil(len(self.image_filenames) / self.batch_size)

    def on_epoch_end(self):
        """Перемешивает индексы в конце каждой эпохи."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Генерирует один батч данных (X, y)."""
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start_idx:end_idx]

        
        batch_image_filenames = [self.image_filenames[i] for i in batch_indexes]

        X = np.empty((len(batch_image_filenames), *self.img_size, self.n_channels), dtype=np.float32)
        y = np.empty((len(batch_image_filenames), *self.img_size, self.n_classes), dtype=np.uint8)

        for i, img_filename in enumerate(batch_image_filenames):
            try:
               
                img_path = os.path.join(self.image_dir, img_filename)
                mask_path = os.path.join(self.mask_dir, img_filename) 

                # Загрузка изображения и маски
                img = imread(img_path)
                mask = imread(mask_path, as_gray=True)

                # Проверка и базовая обработка
                if img is None: raise ValueError(f"imread не смог прочитать изображение: {img_path}")
                if mask is None: raise ValueError(f"imread не смог прочитать маску: {mask_path}")

                if len(img.shape) == 2: img = np.stack((img,)*3, axis=-1)
                elif img.shape[2] == 4: img = img[:,:,:3]
                if img.shape[2] != self.n_channels: raise ValueError(f"Неожиданное кол-во каналов")

                mask = (mask > 0).astype(np.uint8) # Бинаризация 0/1
                mask = np.expand_dims(mask, axis=-1) # -> (H, W, 1)

                # 1. Аугментация
                sample = {'image': img, 'mask': mask}
                if self.augmentation:
                    sample = self.augmentation(**sample)

                # 2. Препроцессинг 
                if self.preprocessing:
                    sample = self.preprocessing(**sample)
                elif sample['image'].shape[:2] != self.img_size:
                     resize_fn = A.Resize(height=self.img_size[0], width=self.img_size[1])
                     sample = resize_fn(**sample)

                X[i,] = sample['image']
                y[i,] = sample['mask']

            except Exception as e:
                print(f"\nОшибка при обработке файла: {img_filename}. Пропуск. Ошибка: {e}")
                X[i,] = np.zeros((*self.img_size, self.n_channels), dtype=np.float32)
                y[i,] = np.zeros((*self.img_size, self.n_classes), dtype=np.uint8)

        return X, y

# --- Функции для создания конвейеров Albumentations ---

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

def get_preprocessing_pipeline(preprocess_input_fn):
    """Возвращает конвейер препроцессинга.
    Args:
        preprocess_input_fn: Функция из segmentation_models (model.preprocess_input)
    """
    
    _transform = []
    if preprocess_input_fn:
        _transform.append(A.Lambda(image=preprocess_input_fn))
    
    return A.Compose(_transform)