import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import segmentation_models as sm
from tqdm import tqdm

# Импорты из локальных модулей
from data_loader import SegmentationDataGenerator, get_validation_augmentation_pipeline, get_preprocessing_pipeline
from metrics_and_losses import get_loss, get_metrics 
from utils import visualize_predictions


# 1. Пути
# !!! ВАЖНО: Замените на ваш реальный путь к ПАПКЕ, содержащей 'input' и 'output' !!!
DATA_ROOT_DIR = '/path/to/your/training_data_folder/' # <--- ИЗМЕНИТЕ ЭТОТ ПУТЬ !!!
IMAGE_DIR_NAME = 'input'
MASK_DIR_NAME = 'output'
# !!! ВАЖНО: Укажите путь к ЛУЧШЕЙ сохраненной модели .h5 !!!
# Найдите файл в папке checkpoints_keras/ после обучения
MODEL_PATH = './checkpoints_keras/best_model_epXX_val_iouX.XXXX.h5' # <--- ИЗМЕНИТЕ ЭТОТ ПУТЬ !!!

# 2. Параметры модели (для препроцессинга и загрузки custom objects)
BACKBONE = 'resnet34'
LOSS_FUNCTION_NAME = 'bce_jaccard' 
METRIC_NAMES = ['iou_score', 'dice_score'] 

# 3. Параметры оценки
IMG_HEIGHT = 256
IMG_WIDTH = 256
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
BATCH_SIZE = 16 
N_CLASSES = 1

# 4. Параметры разделения данных 
TRAIN_VAL_SPLIT = 0.8
RANDOM_STATE = 42
USE_VALIDATION_AS_TEST = True

# 5. Параметры визуализации
NUM_VIS_EXAMPLES = 5

# --- Подготовка ---
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"Segmentation Models version: {sm.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus: print(f"Доступно GPU: {len(gpus)}")
else: print("GPU не найдено, используется CPU.")


# --- Загрузка тестовых ID ---
print("Загрузка списка тестовых файлов...")
image_dir = os.path.join(DATA_ROOT_DIR, IMAGE_DIR_NAME)
mask_dir = os.path.join(DATA_ROOT_DIR, MASK_DIR_NAME)

if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
     print(f"Ошибка: Папки с данными не найдены: {image_dir}, {mask_dir}")
     exit()

try:
    all_image_filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
    all_ids_final = []
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

# Получаем тестовые ID (валидационные)
_, test_ids = train_test_split(all_ids_final, train_size=TRAIN_VAL_SPLIT, random_state=RANDOM_STATE)
print(f"Используется {len(test_ids)} сэмплов для оценки (валидационный набор).")


# --- Препроцессинг и Генератор ---
print("Создание тестового генератора...")
preprocess_input = sm.get_preprocessing(BACKBONE)
test_augmentation = get_validation_augmentation_pipeline(IMG_HEIGHT, IMG_WIDTH)
preprocessing_pipeline = get_preprocessing_pipeline(preprocess_input)

test_generator = SegmentationDataGenerator(
    image_dir=image_dir,
    mask_dir=mask_dir,
    image_ids=test_ids, 
    batch_size=BATCH_SIZE,
    img_size=(IMG_HEIGHT, IMG_WIDTH),
    n_classes=N_CLASSES,
    augmentation=test_augmentation,
    preprocessing=preprocessing_pipeline,
    shuffle_on_epoch_end=False 
)

# --- Загрузка Модели ---
print(f"Загрузка обученной модели из: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"Ошибка: Файл модели не найден: {MODEL_PATH}")
    exit()

# Определяем custom_objects для лосса и метрик
custom_objects = {}
try:
    loss_func_obj = get_loss(LOSS_FUNCTION_NAME)
    custom_objects[getattr(loss_func_obj, '__name__', LOSS_FUNCTION_NAME)] = loss_func_obj
except ValueError as e: print(f"Предупреждение: Не удалось найти лосс '{LOSS_FUNCTION_NAME}': {e}")
try:
    metrics_list_objs = get_metrics(METRIC_NAMES)
    for metric_obj in metrics_list_objs:
         metric_name_in_keras = getattr(metric_obj, '__name__', metric_obj)
         custom_objects[metric_name_in_keras] = metric_obj
except ValueError as e: print(f"Предупреждение: Не удалось найти метрику из '{METRIC_NAMES}': {e}")

print(f"Используемые custom_objects для загрузки: {list(custom_objects.keys())}")

try:
    model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=True)
    print("Модель успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели Keras: {e}")
    print("Убедитесь, что custom_objects содержат правильные объекты потерь/метрик.")
    exit()

# --- Оценка Модели ---
print(f"\nОценка модели на {len(test_ids)} тестовых сэмплах...")
results = model.evaluate(test_generator, verbose=1)

print("\n--- Результаты Оценки ---")
if results:
     print(f"Test Loss: {results[0]:.5f}")
     metric_index = 1
     for name in METRIC_NAMES: 
         internal_keras_name = None

         for keras_metric_name in model.metrics_names[1:]: 
             if name in keras_metric_name: 
                  internal_keras_name = keras_metric_name
                  break
         if internal_keras_name and metric_index < len(results):
             print(f"Test {name} (as {internal_keras_name}): {results[metric_index]:.5f}")
             metric_index += 1
         elif metric_index < len(results):
             # Если не нашли по имени, выводим как есть
             print(f"Test metric_{metric_index} (expected {name}?): {results[metric_index]:.5f}")
             metric_index +=1
         else:
              print(f"Не найдено значение для метрики {name} в результатах evaluate.")
else:
     print("Ошибка: model.evaluate не вернул результаты.")


# --- Визуализация Предсказаний ---
print(f"\nГенерация {NUM_VIS_EXAMPLES} примеров предсказаний...")
try:
    vis_batch_x, vis_batch_y = test_generator.__getitem__(0)
except Exception as e:
     print(f"Ошибка при получении батча из генератора для визуализации: {e}")
     vis_batch_x, vis_batch_y = None, None

if vis_batch_x is not None and vis_batch_y is not None and vis_batch_x.size > 0:
    # Делаем предсказания
    vis_batch_pred = model.predict(vis_batch_x)

    num_to_show = min(NUM_VIS_EXAMPLES, vis_batch_x.shape[0])
    print(f"Отображение первых {num_to_show} примеров...")
    for i in range(num_to_show):
        image = vis_batch_x[i]
        true_mask = vis_batch_y[i]
        predicted_mask = vis_batch_pred[i]

        save_filename = f"prediction_example_{i}.png"
        visualize_predictions(
            image,
            true_mask,
            predicted_mask,
            save_path=save_filename
        )
else:
     print("Не удалось получить батч для визуализации.")

print("\n--- Скрипт evaluate.py завершен ---")