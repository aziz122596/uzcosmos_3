Этот проект реализует модель семантической сегментации (U-Net с бэкбоном EfficientNet/ResNet) с использованием Keras и библиотеки `segmentation-models` для обнаружения дорог на аэроснимках.

## Структура проекта

- `train.py`: Основной скрипт для обучения и валидации модели.
- `evaluate.py`: Скрипт для оценки обученной модели на тестовом наборе и визуализации предсказаний.
- `data_loader.py`: Содержит класс `SegmentationDataGenerator` (Keras Sequence) для загрузки данных и функции для аугментации/препроцессинга.
- `model_builder.py`: Содержит функцию `build_unet_model` для создания архитектуры сегментации (U-Net).
- `metrics_and_losses.py`: Определяет или импортирует функции потерь и метрик из `segmentation_models`.
- `utils.py`: Содержит вспомогательные функции (построение графиков, визуализация).
- `requirements.txt`: Список необходимых Python библиотек (для Keras/TensorFlow).
- `README.md`: Этот файл.
- `.gitignore`: Определяет файлы и папки, игнорируемые Git.
- `checkpoints_keras/`: Папка (создается `train.py`) для сохранения лучшей модели (`.h5` файлы).
- `logs_keras/`: Папка (создается `train.py`) для логов TensorBoard.
- `training_history_keras.png`: Графики лосса/метрик обучения (создается `train.py`).
- `prediction_example_*.png`: Примеры визуализации предсказаний (создается `evaluate.py`).

## Установка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone [https://github.com/aziz122596/uzcosmos_3.git](https://github.com/aziz122596/uzcosmos_3.git) 
    cd uzcosmos_3
    ```

2.  **Скачайте датасет:**
    - **Пользователь должен самостоятельно найти и скачать** один из датасетов: Massachusetts Roads Dataset или DeepGlobe Road Extraction Dataset.
    - **Ссылки для поиска:**
        - Massachusetts Roads: [https://www.cs.toronto.edu/~vmnih/data/](https://www.cs.toronto.edu/~vmnih/data/)
        - DeepGlobe Roads: [http://deepglobe.org/challenge.html](http://deepglobe.org/challenge.html) или [https://competitions.codalab.org/competitions/18467](https://competitions.codalab.org/competitions/18467) (может требовать регистрации).
    - Распакуйте архив.
    - **Убедитесь**, что у вас есть папка, содержащая подпапки `input` (с `.png` изображениями) и `output` (с `.png` масками). Запомните **полный путь** к этой основной папке (той, что содержит `input` и `output`).

3.  **Настройте пути в скриптах:**
    - Откройте файлы `train.py` и `evaluate.py`.
    - Найдите переменную `DATA_ROOT_DIR` и **замените** `/path/to/your/training_data_folder/` на **реальный путь** к вашей папке, содержащей подпапки `input` и `output`.
    - Убедитесь, что `IMAGE_DIR_NAME = 'input'` и `MASK_DIR_NAME = 'output'`.
    - В `evaluate.py` найдите `MODEL_PATH` и подготовьтесь указать путь к файлу `.h5` лучшей модели после обучения.

4.  **Создайте и активируйте виртуальное окружение (рекомендуется):**
    ```bash
    python3 -m venv venv 
    # Linux/macOS:
    source venv/bin/activate
    # Windows:
    # venv\Scripts\activate
    ```

5.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```
    *Важное примечание о версиях: В `requirements.txt` указаны **конкретные версии** TensorFlow и Keras (например, 2.11), так как библиотека `segmentation-models` несовместима с Keras 3. Убедитесь, что установка прошла успешно и используются правильные версии. Установка `tensorflow` может потребовать специфических шагов для поддержки GPU (CUDA, cuDNN).*

## Использование

1.  **Обучение модели:**
    Убедитесь, что путь `DATA_ROOT_DIR` в `train.py` настроен правильно. Запустите обучение:
    ```bash
    python train.py
    ```
    - Лучшая модель будет сохранена в папку `checkpoints_keras/` (имя файла будет включать эпоху и метрику).
    - Графики обучения будут сохранены в `training_history_keras.png`.
    - Логи для TensorBoard будут в папке `logs_keras/`.

2.  **Оценка модели:**
    - **Найдите имя лучшей модели** (`.h5` файл) в папке `checkpoints_keras/`.
    - **Укажите этот полный путь** в переменной `MODEL_PATH` в файле `evaluate.py`.
    - Убедитесь, что `DATA_ROOT_DIR` в `evaluate.py` указан верно.
    - Запустите оценку:
    ```bash
    python evaluate.py
    ```
    - Метрики (Loss, IoU, Dice) будут выведены в консоль.
    - Примеры предсказаний будут сохранены в файлы `prediction_example_*.png`.

## Подход

- **Фреймворк:** Keras (версия 2.x) / TensorFlow (версия 2.x).
- **Модель:** U-Net с предобученным энкодером (EfficientNetB0/ResNet) из библиотеки `segmentation-models`.
- **Датасет:** Massachusetts Roads / DeepGlobe Roads (требуется ручная загрузка и настройка путей).
- **Загрузка данных:** Кастомный генератор `SegmentationDataGenerator` (Keras Sequence) из `data_loader.py`.
- **Предобработка:** Нормализация с помощью `segmentation_models.get_preprocessing`.
- **Аугментация:** Геометрические и яркостные аугментации (`albumentations`) из `data_loader.py`.
- **Функция потерь:** Комбинированная BCE + Jaccard/Dice Loss из `metrics_and_losses.py`.
- **Метрики:** IoU (Jaccard) и F1 (Dice) из `metrics_and_losses.py`.
- **Обучение:** `model.fit` с колбэками Keras (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard) в `train.py`.
- **Оценка:** Выполняется в `evaluate.py` на тестовом (валидационном) наборе с визуализацией.

## Результаты (Пример)

*Заполните этот раздел после получения реальных результатов*
- **Test Loss:** ...
- **Test iou_score:** ...
- **Test dice_score:** ...

*(Сюда можно добавить изображение `prediction_example_0.png`)*

## Возможные улучшения

- Подбор гиперпараметров (бэкбон, LR, батч, лосс, аугментации).
- Использование TTA (Test Time Augmentation).
- Поиск оптимального порога бинаризации для масок.
- Пост-обработка масок (морфологические операции).
- Обучение на плитках (Tiling) для изображений высокого разрешения.
- Использование конфигурационных файлов для параметров.