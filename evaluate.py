import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

# Импорты из локальных модулей
from dataset import RoadDataset, get_validation_augmentation_pipeline, get_preprocessing_pipeline
from model import build_model
from metrics import iou_score, dice_coeff
from utils import visualize_predictions

# --- Конфигурация ---
DATA_ROOT_DIR = '/path/to/your/training_data_folder/' # <--- ИЗМЕНИТЕ ЭТОТ ПУТЬ !!!
IMAGE_DIR_NAME = 'input'
MASK_DIR_NAME = 'output'

ENCODER = 'resnet34'
ENCODER_WEIGHTS = None # Веса загружаем из файла
NUM_CLASSES = 1
ACTIVATION = 'sigmoid' # Должен совпадать с train.py

# Параметры оценки
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16 # Можно увеличить для оценки
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './checkpoints_pytorch/best_model.pth' # <--- ИЗМЕНИТЕ ЭТОТ ПУТЬ 
TRAIN_VAL_SPLIT = 0.8
RANDOM_STATE = 42
USE_VALIDATION_AS_TEST = True # Используем val набор для оценки
NUM_VIS_EXAMPLES = 5

print(f"Используемое устройство: {DEVICE}")

# --- Подготовка тестовых данных ---
print("Подготовка тестовых (валидационных) данных...")
image_dir_full = os.path.join(DATA_ROOT_DIR, IMAGE_DIR_NAME)
mask_dir_full = os.path.join(DATA_ROOT_DIR, MASK_DIR_NAME)

if not os.path.isdir(image_dir_full) or not os.path.isdir(mask_dir_full):
     print(f"Ошибка: Папки с данными не найдены: {image_dir_full}, {mask_dir_full}")
     exit()

try:
    all_image_filenames = sorted([f for f in os.listdir(image_dir_full) if f.lower().endswith('.png')])
    all_ids_final = []
    for img_filename in all_image_filenames:
        mask_filepath = os.path.join(mask_dir_full, img_filename)
        if os.path.exists(mask_filepath):
             all_ids_final.append(img_filename) # Полные имена
except FileNotFoundError: print(f"Ошибка чтения файлов."); exit()

if not all_ids_final: print(f"Ошибка: Не найдены пары изображений/масок."); exit()

# Получаем тестовые (валидационные) ID
_, test_ids = train_test_split(all_ids_final, train_size=TRAIN_VAL_SPLIT, random_state=RANDOM_STATE)
print(f"Используется {len(test_ids)} сэмплов для оценки.")

# --- Препроцессинг и Генератор ---
# Важно: используем тот же энкодер/веса для получения функции препроцессинга, что и при обучении
try:
    preprocessing_fn_eval = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet' if ENCODER_WEIGHTS == 'imagenet' else None)
except: preprocessing_fn_eval = None

test_augmentation = get_validation_augmentation_pipeline(IMG_HEIGHT, IMG_WIDTH)
preprocessing_pipeline = get_preprocessing_pipeline(preprocessing_fn_eval) # Включает ToTensorV2

test_dataset = RoadDataset(
    image_dir_full, mask_dir_full, test_ids, # Передаем полные имена
    transform=test_augmentation, # Только ресайз
    preprocessing=preprocessing_pipeline
)

num_workers = min(os.cpu_count() // 2 if os.cpu_count() else 0, 4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
print("Тестовый датасет и загрузчик созданы.")

# --- Загрузка Модели ---
print(f"Загрузка обученной модели из: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"Ошибка: Файл модели не найден: {MODEL_PATH}"); exit()

# Создаем модель с той же архитектурой
model = build_model(ENCODER, encoder_weights=None, num_classes=NUM_CLASSES, activation=ACTIVATION)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Переводим в режим оценки
    print("Модель успешно загружена.")
except Exception as e: print(f"Ошибка при загрузке весов модели: {e}"); exit()

# --- Оценка на тестовом наборе ---
print("Запуск оценки на тестовом наборе...")
test_iou_total = 0.0
test_dice_total = 0.0
num_batches_processed = 0

# Словари для хранения предсказаний для визуализации
vis_images = []
vis_true_masks = []
vis_pred_masks = []

with torch.no_grad():
    test_loop = tqdm(test_loader, desc="Testing", leave=True)
    for i, (images, masks) in enumerate(test_loop):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE) # masks shape: (B, 1, H, W), float32

        outputs = model(images) # outputs shape: (B, 1, H, W), float32 (вероятности)

        # Считаем метрики
        batch_iou = iou_score(outputs, masks)
        batch_dice = dice_coeff(outputs, masks)

        if not torch.isnan(batch_iou) and not torch.isnan(batch_dice):
             test_iou_total += batch_iou.item()
             test_dice_total += batch_dice.item()
             num_batches_processed += 1

             # Сохраняем первый батч для визуализации (если он корректный)
             if i == 0 and NUM_VIS_EXAMPLES > 0:
                 vis_images.append(images.cpu())
                 vis_true_masks.append(masks.cpu())
                 vis_pred_masks.append(outputs.cpu()) # Сохраняем вероятности

        else:
             print(f"Предупреждение: NaN в метриках батча {i}. Пропуск.")

        test_loop.set_postfix(iou=f"{batch_iou:.4f}", dice=f"{batch_dice:.4f}")

# Рассчитываем средние метрики
if num_batches_processed > 0:
    final_test_iou = test_iou_total / num_batches_processed
    final_test_dice = test_dice_total / num_batches_processed
    print("\n--- Результаты на тестовом (валидационном) наборе ---")
    print(f"Test IoU: {final_test_iou:.4f}")
    print(f"Test Dice: {final_test_dice:.4f}")
else:
     print("\n--- Не удалось рассчитать метрики ---")

# --- Визуализация Результатов ---
print("\nСоздание визуализации предсказаний...")
if vis_images:
    # Объединяем тензоры из сохраненных батчей (здесь только первый)
    vis_images_cat = torch.cat(vis_images)
    vis_true_masks_cat = torch.cat(vis_true_masks)
    vis_pred_masks_cat = torch.cat(vis_pred_masks)

    visualize_predictions(
        vis_images_cat,
        vis_true_masks_cat,
        vis_pred_masks_cat, # Передаем вероятности
        num_examples=NUM_VIS_EXAMPLES,
        filename="prediction_examples_pytorch.png"
    )
else:
    print("Не удалось собрать примеры для визуализации.")

print("\n--- Скрипт evaluate.py завершен ---")