import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp
# train_test_split здесь не нужен, т.к. используем все файлы из папки testing
# from sklearn.model_selection import train_test_split

# Импорты из локальных модулей
from dataset import RoadDataset, get_validation_augmentation_pipeline, get_preprocessing_pipeline
from model import build_model
from metrics import iou_score, dice_coeff
from utils import visualize_predictions

# --- Конфигурация ---
# !!! Укажите правильные пути и параметры !!!

# 1. Пути
# !!! ВАЖНО: Укажите путь к ПАПКЕ 'testing', созданной парсером !!!
DATA_ROOT_DIR = './road_segmentation/testing/' # <--- ПРОВЕРЬТЕ ЭТОТ ПУТЬ !!!
IMAGE_DIR_NAME = 'input'
MASK_DIR_NAME = 'output'
# !!! ВАЖНО: Укажите путь к ЛУЧШЕЙ сохраненной модели .pth !!!
MODEL_PATH = './checkpoints_pytorch/best_model.pth' # <--- ПРОВЕРЬТЕ ЭТОТ ПУТЬ !!!

# 2. Параметры модели (должны совпадать с train.py)
ENCODER = 'resnet34'
ENCODER_WEIGHTS = None # Веса загружаем из файла
NUM_CLASSES = 1
ACTIVATION = 'sigmoid'

# 3. Параметры оценки
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Параметры визуализации
NUM_VIS_EXAMPLES = 5

# --- Подготовка ---
print(f"Используемое устройство: {DEVICE}")

# --- Загрузка тестовых ID (АДАПТИРОВАНО под папку testing и img-*.png) ---
print("Загрузка списка тестовых файлов...")
image_dir_full = os.path.join(DATA_ROOT_DIR, IMAGE_DIR_NAME)
mask_dir_full = os.path.join(DATA_ROOT_DIR, MASK_DIR_NAME)

if not os.path.isdir(image_dir_full) or not os.path.isdir(mask_dir_full):
     print(f"Ошибка: Папки с тестовыми данными не найдены: {image_dir_full}, {mask_dir_full}")
     exit()

try:
    # Получаем ПОЛНЫЕ ИМЕНА файлов img-*.png из папки input
    all_image_filenames = sorted([
        f for f in os.listdir(image_dir_full)
        if f.lower().startswith('img-') and f.lower().endswith('.png')
    ])
    # Проверяем наличие маски с таким же именем в папке output
    test_ids = []
    print(f"Найдено {len(all_image_filenames)} png файлов в {IMAGE_DIR_NAME}. Проверка масок...")
    for img_filename in all_image_filenames:
        mask_filepath = os.path.join(mask_dir_full, img_filename)
        if os.path.exists(mask_filepath):
             test_ids.append(img_filename) # Добавляем полное имя файла

except FileNotFoundError: print(f"Ошибка чтения файлов."); exit()

if not test_ids: print(f"Ошибка: Не найдены пары изображений/масок для оценки."); exit()
print(f"Используется {len(test_ids)} сэмплов для оценки из папки {DATA_ROOT_DIR}.")


# --- Препроцессинг и Генератор ---
print("Создание тестового генератора...")
# Используем тот же энкодер, что при обучении, для получения функции препроцессинга
try:
    preprocessing_fn_eval = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet' if ENCODER_WEIGHTS == 'imagenet' else None)
except: preprocessing_fn_eval = None

test_augmentation = get_validation_augmentation_pipeline(IMG_HEIGHT, IMG_WIDTH)
preprocessing_pipeline = get_preprocessing_pipeline(preprocessing_fn_eval)

test_dataset = RoadDataset(
    image_dir_full, mask_dir_full, test_ids, # Полные имена файлов
    transform=test_augmentation, # Только ресайз
    preprocessing=preprocessing_pipeline
)

num_workers = min(os.cpu_count() // 2 if os.cpu_count() else 0, 4)
# drop_last=False для оценки, чтобы обработать все сэмплы
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, drop_last=False)
print("Тестовый датасет и загрузчик созданы.")


# --- Загрузка Модели ---
print(f"Загрузка обученной модели из: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"Ошибка: Файл модели не найден: {MODEL_PATH}"); exit()

model = build_model(ENCODER, encoder_weights=None, num_classes=NUM_CLASSES, activation=ACTIVATION)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Модель успешно загружена.")
except Exception as e: print(f"Ошибка при загрузке весов модели: {e}"); exit()

# --- Оценка на тестовом наборе ---
print("Запуск оценки на тестовом наборе...")
test_iou_total = 0.0
test_dice_total = 0.0
num_batches_processed = 0

vis_images = []
vis_true_masks = []
vis_pred_masks = []

with torch.no_grad():
    test_loop = tqdm(test_loader, desc="Evaluating", leave=True)
    for i, (images, masks) in enumerate(test_loop):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        outputs = model(images)

        batch_iou = iou_score(outputs, masks)
        batch_dice = dice_coeff(outputs, masks)

        if not torch.isnan(batch_iou) and not torch.isnan(batch_dice):
             test_iou_total += batch_iou.item()
             test_dice_total += batch_dice.item()
             num_batches_processed += 1

             # Сохраняем первый батч для визуализации
             if i == 0 and NUM_VIS_EXAMPLES > 0:
                 vis_images.append(images.cpu())
                 vis_true_masks.append(masks.cpu())
                 vis_pred_masks.append(outputs.cpu())
        else:
             print(f"Предупреждение: NaN в метриках батча {i}.")

        test_loop.set_postfix(iou=f"{batch_iou:.4f}", dice=f"{batch_dice:.4f}")

if num_batches_processed > 0:
    final_test_iou = test_iou_total / num_batches_processed
    final_test_dice = test_dice_total / num_batches_processed
    print("\n--- Результаты на тестовом наборе ---")
    print(f"Test IoU: {final_test_iou:.4f}")
    print(f"Test Dice: {final_test_dice:.4f}")
else:
     print("\n--- Не удалось рассчитать метрики ---")

# --- Визуализация Результатов ---
print("\nСоздание визуализации предсказаний...")
if vis_images:
    vis_images_cat = torch.cat(vis_images)
    vis_true_masks_cat = torch.cat(vis_true_masks)
    vis_pred_masks_cat = torch.cat(vis_pred_masks)
    visualize_predictions(
        vis_images_cat,
        vis_true_masks_cat,
        vis_pred_masks_cat,
        num_examples=NUM_VIS_EXAMPLES,
        filename="prediction_examples_pytorch.png"
    )
else:
    print("Не удалось собрать примеры для визуализации.")

print("\n--- Скрипт evaluate.py завершен ---")