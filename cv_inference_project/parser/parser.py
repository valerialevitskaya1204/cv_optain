#!/usr/bin/env python3
import os
import re
import sys
import requests
import pandas as pd
import logging
from urllib.parse import urlparse
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseDownload
from datetime import datetime
import requests
import re
import os
from urllib.parse import urlencode, urlparse, parse_qs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

GOOGLE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
GOOGLE_CREDENTIALS_FILE = r'creds\credentials.json'
YANDEX_API_URL = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}'
EXCEL_FILE = r'cv_inference_project\parser\датасет.xlsx'
DOWNLOAD_FOLDER = 'downloads'

def setup_folders():
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def get_google_drive_service():
    creds = None

    flow = InstalledAppFlow.from_client_secrets_file(
                GOOGLE_CREDENTIALS_FILE,
                GOOGLE_SCOPES)
    creds = flow.run_local_server(port=0)
    
    return build('drive', 'v3', credentials=creds)

def download_yandex_file(url, folder_path):
    try:
        def normalize_yandex_link(link):
            return re.sub(r"https://(?:disk\.)?360\.yandex\.ru", "https://disk.yandex.ru", link)

        def get_download_url(public_link):
            api_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
            full_url = api_url + urlencode({'public_key': public_link})
            response = requests.get(full_url)
            response.raise_for_status()
            return response.json()['href']

        def extract_filename_from_url(href):
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            return qs.get('filename', ['downloaded_file.bin'])[0]

        def download_file(href, dest_path):
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

        url = normalize_yandex_link(url)
        href = get_download_url(url)
        filename = extract_filename_from_url(href)
        dest_path = os.path.join(folder_path, filename)
        download_file(href, dest_path)

        logger.info(f"Successfully downloaded Yandex Disk file: {dest_path}")
        return True

    except Exception as e:
        logger.error(f"Error downloading Yandex Disk file: {e}")
        return False

def download_google_file(url, folder_path):
    try:
        service = get_google_drive_service()
        
        file_id = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url, re.IGNORECASE)
        if not file_id:
            logger.error(f"Invalid Google Drive URL: {url}")
            return False
        file_id = file_id.group(1)
        
        file_metadata = service.files().get(fileId=file_id).execute()
        filename = file_metadata.get('name', f"google_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        filepath = os.path.join(folder_path, filename)
        
        request = service.files().get_media(fileId=file_id)
        with open(filepath, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.info(f"Download progress: {int(status.progress() * 100)}%")
        
        logger.info(f"Successfully downloaded Google Drive file: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading Google Drive file: {e}")
        return False

def process_download(url, folder_path):
    if 'yandex' in url.lower():
        return download_yandex_file(url, folder_path)
    elif 'google' in url.lower():
        return download_google_file(url, folder_path)
    else:
        logger.error(f"Unsupported URL type: {url}")
        return False

def sanitize_folder_name(name):
    return re.sub(r'[<>:"/\\|?*]', '_', str(name).strip())

def process_excel_file():
    try:
        df = pd.read_excel(EXCEL_FILE)
        
        for _, row in df.iterrows():
            
            user_login = row.get('login', '')
            links_text = row.get('Ссылка/ссылки на облако', '')
            comment = row.get('Комментарий (если что-то пошло не так - впишите сюда что и когда происходило)', '')
            
            if pd.isna(links_text) or not str(links_text).strip():
                logger.info(f"No links found for {user_login}")
                continue
            
            base_folder = os.path.join(DOWNLOAD_FOLDER, sanitize_folder_name(user_login))
            if pd.notna(comment) and str(comment).strip():
                subfolder = sanitize_folder_name(comment)
                folder_path = os.path.join(base_folder, subfolder)
            else:
                folder_path = base_folder
            
            os.makedirs(folder_path, exist_ok=True)
            
            urls = re.findall(r'(https?://[^\s"]+)', str(links_text))
            if not urls:
                logger.info(f"No valid URLs found for {user_login}")
                continue
            
            logger.info(f"Processing {len(urls)} files for {user_login}")
            for i, url in enumerate(urls, 1):
                logger.info(f"Downloading file {i}/{len(urls)}: {url}")
                if not process_download(url, folder_path):
                    logger.error(f"Failed to download: {url}")
                
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")
        sys.exit(1)

def main():
    setup_folders()
    
    if not os.path.exists(EXCEL_FILE):
        logger.error(f"Excel file not found: {EXCEL_FILE}")
        sys.exit(1)
    
    if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
        logger.warning("Google credentials file not found. Only Yandex downloads will work.")
    
    process_excel_file()
    logger.info("Processing completed")

if __name__ == "__main__":
    main()