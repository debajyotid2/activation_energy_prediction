"""Downloads dataset from Zenodo."""
from pathlib import Path
import concurrent.futures
import functools
import logging
import urllib.request
import re

logging.basicConfig(format="%(asctime)s-%(levelname)s-%(thread)d : %(message)s",
                    level=logging.DEBUG)

DATA_URLS = [
        "https://zenodo.org/record/3715478/files/b97d3.csv?download=1",
        "https://zenodo.org/record/3715478/files/b97d3.tar.gz?download=1",
        "https://zenodo.org/record/3715478/files/ts_with_dup_b97d3.tar.gz?download=1",
        "https://zenodo.org/record/3715478/files/ts_with_dup_wb97xd3.tar.gz?download=1",
        "https://zenodo.org/record/3715478/files/wb97xd3.csv?download=1",
        "https://zenodo.org/record/3715478/files/wb97xd3.tar.gz?download=1"
        ]
DATA_DIR = Path("../data")

def download_data(url: str, data_dir: Path) -> None:
    pattern = re.compile(r"/([a-zA-Z0-9\.]+)\?")
    filepath = data_dir / pattern.findall(url)[0]
    
    if not filepath.exists():
        urllib.request.urlretrieve(url, filepath) 
        logging.info(f"{filepath} downloaded.")
    else:
        logging.warning(f"{filepath} already exists.")

def main() -> None:
    urls = DATA_URLS

    downloader = functools.partial(download_data, data_dir=DATA_DIR)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        result = executor.map(downloader, urls)

if __name__ == "__main__":
    main()
