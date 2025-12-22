#!/usr/bin/env python3
"""
Crawl 200 bài viết mới nhất từ fireant.vn/bai-viet
Cứ 2 tiếng chạy 1 lần và lưu ra file mới:
  /workspace/ducdm/API-Gemini/download-data/url-fireant-1.txt
  /workspace/ducdm/API-Gemini/download-data/url-fireant-2.txt
  ...
"""

import requests
from bs4 import BeautifulSoup
import re
import os
import time
from datetime import datetime
import logging

logging.basicConfig(
    filename='./crawl-fireant.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Starting the data scraping process...")

# ========== CẤU HÌNH ==========
BASE_URL = "https://fireant.vn/bai-viet"
OUTPUT_DIR = "./OutputText"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FireAntLinkCollector/1.0; +https://yourdomain.example)"
}
REQUEST_DELAY = 1.0  # giây giữa các request
MAX_LINKS = 200
ARTICLE_REGEX = re.compile(r"^https?://(?:www\.)?fireant\.vn/bai-viet/.+")
RUN_INTERVAL = 60 * 60 / 6 * 5  # 1 tiếng = 3600 giây -> 48 minutes
# ==============================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def next_output_path():
    """Tự động tăng số file: url-fireant-1.txt, 2.txt, 3.txt,..."""
    ensure_dir(OUTPUT_DIR)
    existing = [
        int(re.search(r"url-fireant-(\d+)\.txt", f).group(1))
        for f in os.listdir(OUTPUT_DIR)
        if re.match(r"url-fireant-\d+\.txt", f)
    ]
    next_idx = max(existing) + 1 if existing else 1
    return os.path.join(OUTPUT_DIR, f"url-fireant-{next_idx}.txt")


def fetch_article_links():
    session = requests.Session()
    session.headers.update(HEADERS)

    links = []
    page = 1

    while len(links) < MAX_LINKS:
        url = f"{BASE_URL}?page={page}"
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"[!] Lỗi tải trang {url}: {resp.status_code}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        found_this_page = 0

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("/bai-viet/"):
                full_url = f"https://fireant.vn{href}"
            elif href.startswith("https://fireant.vn/bai-viet/"):
                full_url = href
            else:
                continue

            if ARTICLE_REGEX.match(full_url) and full_url not in links:
                links.append(full_url)
                found_this_page += 1

        if found_this_page == 0:
            # không còn bài mới, dừng
            break

        print(f"Đã lấy {len(links)} link sau trang {page}")
        page += 1
        time.sleep(REQUEST_DELAY)

    return links[:MAX_LINKS]


def run_once():
    print(f"\n===== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    print("Đang thu thập 200 bài viết mới nhất từ fireant.vn...")

    links = fetch_article_links()

    out_path = next_output_path()
    with open(out_path, "w", encoding="utf-8") as f:
        for link in links:
            f.write(link + "\n")

    print(f"✅ Đã lưu {len(links)} link vào: {out_path}\n")
    if links:
        print("15 link đầu tiên:")
        print(links[:15])


def main():
    while True:
        run_once()
        print(f"⏳ Sẽ chạy lại sau {RUN_INTERVAL / 3600:.1f} tiếng...")
        time.sleep(RUN_INTERVAL)


if __name__ == "__main__":
    main()
