import requests
import re
import os
from urllib.parse import urlencode, urlparse, parse_qs

def normalize_yandex_link(link):
    """Преобразует превью и короткие ссылки в /d/-формат"""
    link = re.sub(r"https://(?:disk\.)?360\.yandex\.ru", "https://disk.yandex.ru", link)

    return link

def get_download_url(public_link):
    """Получает временный href для скачивания файла"""
    api_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    full_url = api_url + urlencode({'public_key': public_link})
    response = requests.get(full_url)
    response.raise_for_status()
    return response.json()['href']

def extract_filename_from_url(href):
    """Извлекает имя файла из href, если оно задано"""
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    return qs.get('filename', ['downloaded_video.bin'])[0]

def download_file(href, dest_path):
    """Скачивает файл с href по частям"""
    with requests.get(href, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    done = int(50 * downloaded / total) if total else 0
                    print(f"\r[{'█' * done:<50}] {downloaded/1024/1024:.2f} MB", end='')

    print(f"\n✅ Файл сохранён: {dest_path}")

# === Запуск ===
public_link = 'https://disk.360.yandex.ru/i/0RQ9U_BnOW1Fdw'  # твоя ссылка
normalized = normalize_yandex_link(public_link)

try:
    href = get_download_url(normalized)
    filename = extract_filename_from_url(href)
    save_path = os.path.join(os.getcwd(), filename)
    download_file(href, save_path)
except Exception as e:
    print(f"❌ Ошибка при скачивании: {e}")
