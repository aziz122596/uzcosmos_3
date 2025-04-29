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
import subprocess # Для wget
import tarfile    # Для tar.gz
from sklearn.model_selection import train_test_split
import time

# Импорты из локальных модулей PyTorch
from dataset import RoadDataset, get_training_augmentation_pipeline, get_validation_augmentation_pipeline, get_preprocessing_pipeline
from model import build_model
from metrics import iou_score, dice_coeff
from utils import plot_training_history # Импортируем нашу функцию

# --- Конфигурация ---

# --- Конфигурация Загрузки Данных (Пример для Massachusetts Roads) ---
DATASET_URL = "https://www.cs.toronto.edu/~vmnih/data/massachusetts_roads_archives.tar.gz"
DOWNLOAD_DIR = "./downloaded_data"
EXTRACT_DIR = "./extracted_data"
EXPECTED_DATASET_FOLDER_NAME = "massachusetts_roads"
DATA_DIR = os.path.join(EXTRACT_DIR, EXPECTED_DATASET_FOLDER_NAME)
# IMAGE_DIR_NAME = 'input'
# MASK_DIR_NAME = 'output'
# Если используете официальную версию Massachusetts Roads:
IMAGE_DIR_NAME = os.path.join("tiff", "images")
MASK_DIR_NAME = os.path.join("tiff", "masks")

# --- Параметры модели ---
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
NUM_CLASSES = 1
ACTIVATION = 'sigmoid' # sigmoid для бинарной классификации с Dice/Jaccard/BCE loss

# --- Параметры обучения ---
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

# --- Подготовка ---
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
print(f"Используемое устройство: {DEVICE}")

# --- Автоматическая Загрузка и Распаковка ---
def download_and_extract_dataset(url, download_path, extract_path, expected_dir):
    archive_filename = os.path.basename(url)
    archive_filepath = os.path.join(download_path, archive_filename)
    expected_dataset_path = os.path.join(extract_path, expected_dir)
    # Проверяем наличие подпапки с изображениями как индикатор
    check_final_dir = os.path.join(expected_dataset_path, IMAGE_DIR_NAME)

    print("--- Проверка наличия датасета ---")
    if os.path.isdir(check_final_dir):
        print(f"Датасет уже найден в: {expected_dataset_path}")
        return True

    print(f"Распакованный датасет не найден. Проверка архива {archive_filename}...")
    if not os.path.exists(archive_filepath):
        print(f"Архив не найден. Загрузка из {url}...")
        try:
            command = ["wget", "-c", "-P", download_path, url]
            print(f"Выполнение команды: {' '.join(command)}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Скачивание успешно завершено: {archive_filepath}")
        except FileNotFoundError: print("Ошибка: 'wget' не найден."); return False
        except subprocess.CalledProcessError as e: print(f"Ошибка скачивания wget: {e.stderr}"); return False
        except Exception as e: print(f"Ошибка скачивания: {e}"); return False
    else: print(f"Архив {archive_filename} уже существует.")

    print(f"Распаковка {archive_filepath} в {extract_path}...")
    try:
        with tarfile.open(archive_filepath, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print("Распаковка завершена.")
        if os.path.isdir(check_final_dir):
            print(f"Датасет успешно распакован: {expected_dataset_path}")
            return True
        else: print(f"Ошибка: Распаковка прошла, но папка не найдена: {check_final_dir}"); return False
    except tarfile.TarError as e: print(f"Ошибка распаковки tar.gz: {e}"); return False
    except Exception as e: print(f"Ошибка распаковки: {e}"); return False

dataset_ready = download_and_extract_dataset(
    DATASET_URL, DOWNLOAD_DIR, EXTRACT_DIR, EXPECTED_DATASET_FOLDER_NAME
)

if not dataset_ready:
    print("\n--- ОСТАНОВКА: Не удалось подготовить датасет. ---")
    exit()

# --- Подготовка данных ---
print("\nПодготовка данных...")
image_dir_full = os.path.join(DATA_DIR, IMAGE_DIR_NAME)
mask_dir_full = os.path.join(DATA_DIR, MASK_DIR_NAME)

if not os.path.isdir(image_dir_full): print(f"Критическая ошибка: Папка изображений не найдена: {image_dir_full}"); exit()
if not os.path.isdir(mask_dir_full): print(f"Критическая ошибка: Папка масок не найдена: {mask_dir_full}"); exit()

try:
    # Получаем ПОЛНЫЕ ИМЕНА ФАЙЛОВ .png (адаптировано под структуру Kaggle/input)
    all_image_filenames = sorted([f for f in os.listdir(image_dir_full) if f.lower().endswith('.png')])
    all_ids_final = []
    print(f"Найдено {len(all_image_filenames)} png файлов в {IMAGE_DIR_NAME}. Проверка масок...")
    for img_filename in all_image_filenames:
        mask_filepath = os.path.join(mask_dir_full, img_filename) # Ищем маску с тем же именем
        if os.path.exists(mask_filepath):
             all_ids_final.append(img_filename)

except FileNotFoundError: print(f"Ошибка чтения файлов."); exit()

if not all_ids_final: print(f"Ошибка: Не найдены пары изображений/масок."); exit()
print(f"Найдено {len(all_ids_final)} пар изображений/масок.")

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
    transform=val_augmentation, # Только ресайз
    preprocessing=preprocessing_pipeline
)

num_workers = min(os.cpu_count() // 2 if os.cpu_count() else 0, 4)
print(f"Используется {num_workers} воркеров для DataLoader.")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
print("Датасеты и загрузчики созданы.")

# --- Модель, Лосс, Оптимизатор ---
print("Создание модели, лосса и оптимизатора...")
model = build_model(ENCODER, ENCODER_WEIGHTS, NUM_CLASSES, ACTIVATION)
model.to(DEVICE)

loss_fn = smp.losses.DiceLoss(mode='binary')
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True) # mode='max' для метрик IoU/Dice
metrics_dict = {'iou': iou_score, 'dice': dice_coeff}
print("Модель, лосс и оптимизатор настроены.")

# --- Цикл обучения и валидации ---
history = {'train_loss': [], 'val_loss': [], f'val_{METRIC_TO_MONITOR}': []} # Добавим другие метрики ниже
best_val_metric = -1.0 # Начинаем с -1 для метрик типа IoU/Dice

print(f"\nНачало обучения на {EPOCHS} эпох...")
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    print(f"\n--- Эпоха {epoch+1}/{EPOCHS} ---")

    # --- Фаза обучения ---
    model.train()
    running_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=True)
    for batch_idx, (images, masks) in enumerate(train_loop):
        images = images.to(DEVICE) # non_blocking убран для совместимости
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        train_loop.set_postfix(loss=f"{avg_loss:.4f}")

    epoch_train_loss = running_loss / len(train_loader)
    history['train_loss'].append(epoch_train_loss)
    print(f"Train Loss: {epoch_train_loss:.4f}")

    # --- Фаза валидации ---
    model.eval()
    val_loss = 0.0
    val_metrics_accum = {name: 0.0 for name in metrics_dict}

    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Eval Epoch {epoch+1}", leave=True)
        for images, masks in val_loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            for name, metric_fn in metrics_dict.items():
                batch_metric = metric_fn(outputs, masks)
                val_metrics_accum[name] += batch_metric.item()

    # Средние значения за эпоху
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_metrics = {name: val_metrics_accum[name] / len(val_loader) for name in metrics_dict}

    history['val_loss'].append(epoch_val_loss)
    metrics_log_str = ""
    for name, value in epoch_val_metrics.items():
         history.setdefault(f'val_{name}', []).append(value) # Добавляем все метрики в историю
         metrics_log_str += f"Val {name.capitalize()}: {value:.4f}, "

    print(f"Val Loss: {epoch_val_loss:.4f}, {metrics_log_str.strip(', ')}")

    # Шаг планировщика по основной метрике
    current_metric_value = epoch_val_metrics[METRIC_TO_MONITOR]
    scheduler.step(current_metric_value)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Текущий Learning Rate: {current_lr:.6f}")

    # Сохраняем лучшую модель
    if current_metric_value > best_val_metric:
        best_val_metric = current_metric_value
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"!!! Модель сохранена как лучшая (по {METRIC_TO_MONITOR}={best_val_metric:.4f}) в {checkpoint_path} !!!")

    epoch_end_time = time.time()
    print(f"Эпоха {epoch+1} завершена за {epoch_end_time - epoch_start_time:.2f} секунд.")

# --- Завершение обучения ---
total_training_time = time.time() - start_time
print(f"\nОбучение завершено за {total_training_time / 60:.2f} минут.")
print(f"Лучшее значение метрики валидации ({METRIC_TO_MONITOR}): {best_val_metric:.4f}")

# --- Построение графиков обучения ---
plot_training_history(history, save_path="training_history_pytorch.png")

print("\n--- Скрипт train.py завершен ---")