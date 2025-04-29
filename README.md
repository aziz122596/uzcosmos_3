Этот проект реализует модель семантической сегментации (U-Net с бэкбоном ResNet34) с использованием PyTorch и библиотеки `segmentation-models-pytorch` для обнаружения дорог на аэроснимках. Данные для обучения и тестирования предполагается загружать с помощью скрипта `download_massroads.py`, который парсит HTML-страницы датасета Massachusetts Roads.

## Структура проекта

- `download_massroads.py`: Скрипт для скачивания изображений и масок датасета Massachusetts Roads путем парсинга HTML-индексов (запускается первым).
- `train.py`: Основной скрипт для обучения и валидации модели на данных, скачанных парсером.
- `evaluate.py`: Скрипт для оценки обученной модели на тестовом наборе (скачанном парсером) и визуализации предсказаний.
- `dataset.py`: Содержит класс `RoadDataset` (PyTorch Dataset) для загрузки данных и функции для аугментации/препроцессинга.
- `model.py`: Содержит функцию `build_model` для создания архитектуры сегментации (U-Net) с помощью `segmentation-models-pytorch`.
- `metrics.py`: Содержит функции для расчета метрик IoU и Dice на тензорах PyTorch.
- `utils.py`: Содержит вспомогательные функции (графики, денормализация, визуализация).
- `requirements.txt`: Список необходимых Python библиотек (для PyTorch и парсера).
- `README.md`: Этот файл.
- `.gitignore`: Определяет файлы и папки, игнорируемые Git.
- `road_segmentation/`: Папка (создается `download_massroads.py`) с подпапками `training` и `testing`, каждая из которых содержит `input` и `output`.
- `checkpoints_pytorch/`: Папка (создается `train.py`) для сохранения лучшей модели (`best_model.pth`).
- `training_history_pytorch.png`: Графики лосса/метрик обучения (создается `train.py`).
- `prediction_examples_pytorch.png`: Примеры визуализации предсказаний (создается `evaluate.py`).

## Установка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone <URL_вашего_репозитория>
    cd <папка_репозитория>
    ```

2.  **Создайте и активируйте виртуальное окружение:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # (или venv\Scripts\activate для Windows)
    ```

3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```
    *Примечание: Установка `torch` может потребовать специфических команд в зависимости от вашей ОС и версии CUDA. См. [официальный сайт PyTorch](https://pytorch.org/).*

## Использование

1.  **Скачайте данные:**
    Запустите скрипт `download_massroads.py`. Он попытается скачать изображения и маски с сайта `cs.toronto.edu` и сохранить их в папку `./road_segmentation/`.
    ```bash
    python download_massroads.py
    ```
    *Убедитесь, что URL в скрипте все еще действительны. Если нет, вам придется найти данные вручную и адаптировать пути в шаге 2.*

2.  **Настройте пути (если нужно):**
    - Скрипты `train.py` и `evaluate.py` по умолчанию ожидают найти данные в `./road_segmentation/training/` и `./road_segmentation/testing/` соответственно.
    - Если скрипт `download_massroads.py` успешно создал эту структуру, изменять пути `DATA_ROOT_DIR` не нужно.
    - Если вы скачали данные вручную или в другую папку, **отредактируйте** переменную `DATA_ROOT_DIR` в `train.py` и `evaluate.py`, указав правильный путь к папкам `training` и `testing`.

3.  **Обучение модели:**
    Запустите скрипт `train.py`.
    ```bash
    python train.py
    ```
    - Лучшая модель будет сохранена в `checkpoints_pytorch/best_model.pth`.
    - Графики обучения будут сохранены в `training_history_pytorch.png`.

4.  **Оценка модели:**
    - Убедитесь, что `DATA_ROOT_DIR` в `evaluate.py` указывает на папку с тестовыми данными (`./road_segmentation/testing/`).
    - Укажите правильный путь к сохраненной модели в `MODEL_PATH` (по умолчанию `./checkpoints_pytorch/best_model.pth`).
    - Запустите оценку:
    ```bash
    python evaluate.py
    ```
    - Метрики IoU и Dice будут выведены в консоль.
    - Примеры предсказаний будут сохранены в `prediction_examples_pytorch.png`.

## Подход
(Аналогично предыдущей PyTorch версии, но уточнить про метод загрузки)
- **Фреймворк:** PyTorch.
- **Модель:** U-Net с предобученным энкодером (ResNet34) из библиотеки `segmentation-models-pytorch`.
- **Датасет:** Massachusetts Roads (загружается скриптом `download_massroads.py` из источника UofT, сохраняется как PNG в папках input/output).
- **Загрузка данных:** Кастомный `RoadDataset` (PyTorch Dataset).
- **Предобработка/Аугментация:** `albumentations`.
- **Функция потерь:** Dice Loss.
- **Метрики:** IoU и Dice Coefficient.
- **Обучение:** Кастомный цикл PyTorch.
