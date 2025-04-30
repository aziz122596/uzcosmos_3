import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time

# Импорты из локальных модулей PyTorch
from dataset import RoadDataset, get_training_augmentation_pipeline, get_validation_augmentation_pipeline, get_preprocessing_pipeline
from model import build_model
from metrics import iou_score, dice_coeff
from utils import plot_training_history

# --- Конфигурация ---

# Папка, содержащая подпапки input/output для ОБУЧЕНИЯ
DATA_ROOT_DIR = './road_segmentation/training/' # <--- Убедитесь, что этот путь ВЕРНЫЙ!
IMAGE_DIR_NAME = 'input'
MASK_DIR_NAME = 'output'
# --------------------------------------------------------------------

# Параметры модели
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
NUM_CLASSES = 1
ACTIVATION = 'sigmoid'

# Параметры обучения
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = './checkpoints_pytorch'
TRAIN_VAL_SPLIT = 0.8
RANDOM_STATE = 42
METRIC_TO_MONITOR = 'dice' # 'iou' или 'dice'

# --- Подготовка Папок ---
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Используемое устройство: {DEVICE}")

# --- Подготовка данных ---
print("\nПодготовка данных...")
image_dir_full = os.path.join(DATA_ROOT_DIR, IMAGE_DIR_NAME)
mask_dir_full = os.path.join(DATA_ROOT_DIR, MASK_DIR_NAME)

# Проверка существования папок
if not os.path.isdir(image_dir_full): print(f"Критическая ошибка: Папка изображений не найдена: {image_dir_full}"); exit()
if not os.path.isdir(mask_dir_full): print(f"Критическая ошибка: Папка масок не найдена: {mask_dir_full}"); exit()

try:
    # Получаем ПОЛНЫЕ ИМЕНА файлов img-*.png из папки input
    all_image_filenames = sorted([
        f for f in os.listdir(image_dir_full)
        if f.lower().startswith('img-') and f.lower().endswith('.png')
    ])
    # Проверяем наличие маски с таким же именем в папке output
    all_ids_final = []
    print(f"Найдено {len(all_image_filenames)} img-*.png файлов в {IMAGE_DIR_NAME}. Проверка масок...")
    for img_filename in all_image_filenames:
        mask_filepath = os.path.join(mask_dir_full, img_filename)
        if os.path.exists(mask_filepath):
             all_ids_final.append(img_filename) # Добавляем полное имя файла

except FileNotFoundError: print(f"Ошибка чтения файлов."); exit()

if not all_ids_final: print(f"Ошибка: Не найдены совпадающие пары изображений/масок (img-*.png)."); exit()
print(f"Найдено {len(all_ids_final)} пар изображений/масок.")

# Разделение на train/validation
train_ids, val_ids = train_test_split(all_ids_final, train_size=TRAIN_VAL_SPLIT, random_state=RANDOM_STATE)
print(f"Разделение: {len(train_ids)} train, {len(val_ids)} validation")

# --- Препроцессинг и Аугментация ---
try:
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    print(f"Используется препроцессинг для {ENCODER}/{ENCODER_WEIGHTS}.")
except:
    print("Предупреждение: Используется стандартная нормализация.")
    preprocessing_fn = None

train_augmentation = get_training_augmentation_pipeline(IMG_HEIGHT, IMG_WIDTH)
val_augmentation = get_validation_augmentation_pipeline(IMG_HEIGHT, IMG_WIDTH)
preprocessing_pipeline = get_preprocessing_pipeline(preprocessing_fn) # Включает ToTensorV2

# --- Создание Датасетов и Загрузчиков ---
train_dataset = RoadDataset(
    image_dir_full, mask_dir_full, train_ids, # Передаем полные имена
    transform=train_augmentation,
    preprocessing=preprocessing_pipeline
)
val_dataset = RoadDataset(
    image_dir_full, mask_dir_full, val_ids, # Передаем полные имена
    transform=val_augmentation,
    preprocessing=preprocessing_pipeline
)

num_workers = min(os.cpu_count() // 2 if os.cpu_count() else 0, 4)
print(f"Используется {num_workers} воркеров для DataLoader.")
# drop_last=True для train_loader может быть полезен, если размер батча не делит набор данных нацело
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
print("Датасеты и загрузчики созданы.")

# --- Модель, Лосс, Оптимизатор ---
print("Создание модели, лосса и оптимизатора...")
model = build_model(ENCODER, ENCODER_WEIGHTS, NUM_CLASSES, ACTIVATION)
model.to(DEVICE)

loss_fn = smp.losses.DiceLoss(mode='binary')
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3) # mode='max' для метрик
metrics_dict = {'iou': iou_score, 'dice': dice_coeff}
print("Модель, лосс и оптимизатор настроены.")

# --- Цикл обучения и валидации ---
history = {'train_loss': [], 'val_loss': [], f'val_{METRIC_TO_MONITOR}': []}
best_val_metric = -1.0

print(f"\nНачало обучения на {EPOCHS} эпох...")
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    print(f"\n--- Эпоха {epoch+1}/{EPOCHS} ---")

    # Обучение
    model.train()
    running_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=True)
    for batch_idx, (images, masks) in enumerate(train_loop):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_loop.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}")
    epoch_train_loss = running_loss / len(train_loader)
    history['train_loss'].append(epoch_train_loss)
    print(f"Train Loss: {epoch_train_loss:.4f}")

    # Валидация
    model.eval()
    val_loss = 0.0
    val_metrics_accum = {name: 0.0 for name in metrics_dict}
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Eval Epoch {epoch+1}", leave=True)
        for images, masks in val_loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
            for name, metric_fn in metrics_dict.items():
                batch_metric = metric_fn(outputs, masks)
                if not torch.isnan(batch_metric): val_metrics_accum[name] += batch_metric.item()
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_metrics = {name: val_metrics_accum[name] / len(val_loader) for name in metrics_dict}
    history['val_loss'].append(epoch_val_loss)
    metrics_log_str = ""
    for name, value in epoch_val_metrics.items():
         history.setdefault(f'val_{name}', []).append(value)
         metrics_log_str += f"Val {name.capitalize()}: {value:.4f}, "
    print(f"Val Loss: {epoch_val_loss:.4f}, {metrics_log_str.strip(', ')}")

    # Шаг планировщика и сохранение лучшей модели
    current_metric_value = epoch_val_metrics[METRIC_TO_MONITOR]
    scheduler.step(current_metric_value)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Текущий Learning Rate: {current_lr:.6f}")
    if current_metric_value > best_val_metric:
        best_val_metric = current_metric_value
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"!!! Модель сохранена как лучшая ({METRIC_TO_MONITOR}={best_val_metric:.4f}) в {checkpoint_path} !!!")

    epoch_end_time = time.time()
    print(f"Эпоха {epoch+1} завершена за {epoch_end_time - epoch_start_time:.2f} секунд.")

# --- Завершение обучения ---
total_training_time = time.time() - start_time
print(f"\nОбучение завершено за {total_training_time / 60:.2f} минут.")
print(f"Лучшее значение метрики валидации ({METRIC_TO_MONITOR}): {best_val_metric:.4f}")

# --- Построение графиков обучения ---
plot_training_history(history, save_path="training_history_pytorch.png")

print("\n--- Скрипт train.py завершен ---")