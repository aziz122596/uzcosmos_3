import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import segmentation_models as sm
from tqdm import tqdm

from data_loader import (SegmentationDataGenerator, get_training_augmentation_pipeline,
                           get_validation_augmentation_pipeline, get_preprocessing_pipeline)
from model_builder import build_unet_model
from metrics_and_losses import get_loss, get_metrics
from utils import plot_training_history

# --- Конфигурация ---
# 1. Пути к данным
DATA_ROOT_DIR = '/path/to/your/training_data_folder/' # <--- ИЗМЕНИТЕ ЭТОТ ПУТЬ !!!
IMAGE_DIR_NAME = 'input'
MASK_DIR_NAME = 'output'

# 2. Параметры модели
BACKBONE = 'resnet34' 
ENCODER_WEIGHTS = 'imagenet'
ENCODER_FREEZE = False
N_CLASSES = 1
ACTIVATION = 'sigmoid'

# 3. Параметры обучения
IMG_HEIGHT = 256
IMG_WIDTH = 256
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
LOSS_FUNCTION_NAME = 'bce_jaccard'
METRIC_NAMES = ['iou_score', 'dice_score']

# 4. Параметры сохранения и колбэков
CHECKPOINT_DIR = './checkpoints_keras'
CHECKPOINT_FILENAME = 'best_model_ep{epoch:02d}_val_iou{val_iou_score:.4f}.h5'
MONITOR_METRIC = 'val_iou_score' 
MONITOR_MODE = 'max'
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
TENSORBOARD_LOG_DIR = './logs_keras'

# 5. Параметры разделения данных
TRAIN_VAL_SPLIT = 0.8
RANDOM_STATE = 42

# --- Подготовка ---
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"Segmentation Models version: {sm.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Доступно GPU: {len(gpus)}")
    except RuntimeError as e: print(e)
else: print("GPU не найдено, используется CPU.")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# --- Загрузка и разделение ID файлов---
print("Загрузка списка файлов...")
image_dir = os.path.join(DATA_ROOT_DIR, IMAGE_DIR_NAME)
mask_dir = os.path.join(DATA_ROOT_DIR, MASK_DIR_NAME)

if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
     print(f"Ошибка: Папки с данными не найдены: {image_dir}, {mask_dir}")
     print(f"Проверьте значение DATA_ROOT_DIR: {DATA_ROOT_DIR}")
     exit()

try:
    all_image_filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
    all_ids_final = []
    print(f"Найдено {len(all_image_filenames)} png файлов в {IMAGE_DIR_NAME}. Проверка масок...")
    for img_filename in all_image_filenames:
        mask_filepath = os.path.join(mask_dir, img_filename)
        if os.path.exists(mask_filepath):
             all_ids_final.append(img_filename) 

except FileNotFoundError:
     print(f"Ошибка: Не удалось прочитать файлы из {image_dir} или {mask_dir}.")
     exit()

if not all_ids_final:
    print(f"Ошибка: Не найдены совпадающие пары изображений/масок.")
    exit()

print(f"Найдено {len(all_ids_final)} пар изображений/масок.")

# Разделение на train/validation
train_ids, val_ids = train_test_split(all_ids_final, train_size=TRAIN_VAL_SPLIT, random_state=RANDOM_STATE)
print(f"Разделение: {len(train_ids)} train, {len(val_ids)} validation")

# --- Препроцессинг и Аугментация ---
preprocess_input = sm.get_preprocessing(BACKBONE)
train_augmentation = get_training_augmentation_pipeline(IMG_HEIGHT, IMG_WIDTH)
val_augmentation = get_validation_augmentation_pipeline(IMG_HEIGHT, IMG_WIDTH)
preprocessing_pipeline = get_preprocessing_pipeline(preprocess_input)

# --- Создание Генераторов ---
print("Создание генераторов данных...")
train_generator = SegmentationDataGenerator(
    image_dir=image_dir,
    mask_dir=mask_dir,
    image_filenames=train_ids, 
    batch_size=BATCH_SIZE,
    img_size=(IMG_HEIGHT, IMG_WIDTH),
    n_classes=N_CLASSES,
    augmentation=train_augmentation,
    preprocessing=preprocessing_pipeline,
    shuffle_on_epoch_end=True
)

val_generator = SegmentationDataGenerator(
    image_dir=image_dir,
    mask_dir=mask_dir,
    image_filenames=val_ids, 
    batch_size=BATCH_SIZE,
    img_size=(IMG_HEIGHT, IMG_WIDTH),
    n_classes=N_CLASSES,
    augmentation=val_augmentation, 
    preprocessing=preprocessing_pipeline,
    shuffle_on_epoch_end=False
)

# --- Создание и Компиляция Модели ---
model = build_unet_model(
    backbone=BACKBONE,
    input_shape=INPUT_SHAPE,
    classes=N_CLASSES,
    activation=ACTIVATION,
    encoder_weights=ENCODER_WEIGHTS,
    encoder_freeze=ENCODER_FREEZE
)

loss_func = get_loss(LOSS_FUNCTION_NAME)
metrics_list = get_metrics(METRIC_NAMES)
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics_list)
print("Модель скомпилирована.")
model.summary()

# --- Колбэки ---
print("Настройка колбэков...")
checkpoint_filepath = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor=MONITOR_METRIC,
        mode=MONITOR_MODE,
        save_best_only=True,
        save_weights_only=False, 
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor=MONITOR_METRIC,
        mode=MONITOR_MODE,
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor=MONITOR_METRIC,
        mode=MONITOR_MODE,
        patience=EARLY_STOPPING_PATIENCE,
        verbose=1,
        restore_best_weights=True 
    ),
    keras.callbacks.TensorBoard(
        log_dir=TENSORBOARD_LOG_DIR,
        histogram_freq=0
    )
]
print("Колбэки настроены.")

# --- Обучение Модели ---
print(f"Начало обучения на {EPOCHS} эпох...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    shuffle=False, 
    verbose=1
)

print("Обучение завершено.")

# --- Сохранение Графика Обучения ---
plot_training_history(history, save_path="training_history_keras.png")

print("\n--- Скрипт train.py завершен ---")