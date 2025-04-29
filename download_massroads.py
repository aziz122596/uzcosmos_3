# code from https://github.com/BBarbosa/tflearn-image-recognition-toolkit/blob/4a0528dcfb206b1e45997f2fbc097aafacfa0fa0/scripts/html_link_parser.py
# Убедитесь, что установлены: pip install beautifulsoup4 html5lib Pillow requests

import re
# import argparse # Не используется в этом фрагменте
import time # Добавим для пауз
from PIL import Image
# from io import BytesIO # Не используется напрямую здесь
from bs4 import BeautifulSoup
# from skimage import io as skio # Лучше использовать Pillow/requests
# from urllib.request import urlopen # Заменим на requests для большей надежности
import requests # Используем requests
from requests.exceptions import RequestException
import os

def html_url_parser(url, save_dir, show=False, wait=False):
    """
    HTML parser to download images from URL.
    Params:\n
    `url` - Image index page url\n
    `save_dir` - Directory to save extracted images\n
    `show` - Show downloaded image (requires GUI environment)\n
    `wait` - Press key to continue executing (for debugging)
    """
    print(f"[INFO] Parsing URL: {url}")
    print(f"[INFO] Saving files to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True) # Создаем папку, если ее нет

    try:
        # Используем requests с таймаутом и обработкой ошибок
        headers = {'User-Agent': 'Mozilla/5.0'} # Некоторые сайты требуют User-Agent
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status() # Проверяем на ошибки HTTP (вроде 404)
        html = response.text # Используем .text для HTML
    except RequestException as e:
        print(f"[EXCEPTION] Failed to fetch HTML from {url}: {e}")
        return # Выходим, если не смогли скачать HTML

    soup = BeautifulSoup(html, "html5lib")

    links_found = soup.find_all("a", href=True)
    print(f"[INFO] Found {len(links_found)} links on the page.")

    download_count = 0
    skip_count = 0
    error_count = 0

    # Используемenumerate, начиная с 1 для ID файла, и берем все ссылки
    for image_id, link in enumerate(links_found, start=1):
        # Не пропускаем первую ссылку, если она ведет на изображение
        href = link.get("href") # Используем .get для безопасности
        if not href or not (href.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))):
            # print(f"[DEBUG] Skipping non-image link: {href}")
            continue # Пропускаем, если ссылка не похожа на изображение

        # Формируем полный URL, если ссылка относительная
        if not href.startswith(('http://', 'https://')):
            from urllib.parse import urljoin
            img_url = urljoin(url, href) # Собираем полный URL
        else:
            img_url = href

        save_filename = f"img-{image_id}.png" # Сохраняем все как PNG
        save_filepath = os.path.join(save_dir, save_filename)

        try:
            if not os.path.isfile(save_filepath):
                print(f"[INFO] Downloading image {image_id}/{len(links_found)} from: {img_url} ... ", end='', flush=True)
                # Скачиваем изображение с помощью requests
                img_response = requests.get(img_url, headers=headers, timeout=30, stream=True)
                img_response.raise_for_status()

                # Открываем и сохраняем с помощью Pillow (более надежно для разных форматов)
                with Image.open(img_response.raw) as image:
                    # Можно добавить конвертацию в RGB перед сохранением, если нужно единообразие
                    # image = image.convert('RGB')
                    image.save(save_filepath, "PNG")
                print("Done.")
                download_count += 1
                if show:
                    try:
                        # Попытка показать изображение (может не работать на сервере без GUI)
                         with Image.open(save_filepath) as img_to_show:
                             img_to_show.show()
                    except Exception as show_e:
                         print(f"[WARNING] Could not show image: {show_e}")

                time.sleep(0.1) # Небольшая пауза, чтобы не перегружать сервер
            else:
                # print(f"Skipped (already exists): {save_filepath}")
                skip_count += 1

        except KeyboardInterrupt:
            print("\n[EXCEPTION] Pressed 'Ctrl+C'")
            break
        except RequestException as img_req_exc:
             print(f"\n[EXCEPTION] Failed to download {img_url}: {img_req_exc}")
             error_count += 1
        except IOError as img_io_exc:
             print(f"\n[EXCEPTION] Failed to open/save image from {img_url}: {img_io_exc}")
             error_count += 1
        except Exception as image_exception:
            print(f"\n[EXCEPTION] Unknown error for {img_url}: {image_exception}")
            error_count += 1
            continue

        if wait:
            try:
                key = input("[INFO] Press any key to continue ('q' to exit)... ")
                if key.lower() == "q":
                    break
            except EOFError: # Если запускается не интерактивно
                 pass


    print(f"\n[INFO] Download process finished.")
    print(f"[SUMMARY] Downloaded: {download_count}, Skipped: {skip_count}, Errors: {error_count}")


# ///////////////////////////////////////////////////
#         Main method
# ///////////////////////////////////////////////////
if __name__ == "__main__":
    # Используем URLы на index.html страницы, содержащие ссылки на файлы
    # Убедитесь, что эти URL все еще работают!
    URL_TRAIN_IMG = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html"
    URL_TRAIN_GT = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html"

    URL_TEST_IMG = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/index.html" # 'valid' используется как test
    URL_TEST_GT = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/index.html" # 'valid' используется как test

    # Определяем базовую папку для сохранения
    BASE_SAVE_DIR = "./road_segmentation"

    # Создаем папки и запускаем парсер для каждого набора
    html_url_parser(url=URL_TRAIN_IMG, save_dir=os.path.join(BASE_SAVE_DIR, "training", "input"))
    html_url_parser(url=URL_TRAIN_GT, save_dir=os.path.join(BASE_SAVE_DIR, "training", "output"))

    html_url_parser(url=URL_TEST_IMG, save_dir=os.path.join(BASE_SAVE_DIR, "testing", "input"))
    html_url_parser(url=URL_TEST_GT, save_dir=os.path.join(BASE_SAVE_DIR, "testing", "output"))

    print("\n[INFO] All downloading tasks are done!")