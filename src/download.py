"""Downloads dataset from Zenodo."""
from pathlib import Path
import concurrent.futures
import functools
import logging
import urllib.request
import re
import tarfile

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
    """
    Downloads files from Grambow dataset into specified data directory.
    """
    pattern = re.compile(r"/([a-zA-Z0-9\.]+)\?")
    filepath = data_dir / pattern.findall(url)[0]
    
    if not filepath.exists():
        if filepath.suffix == ".gz" and (data_dir / filepath.stem).exists():
            logging.warning(f"{filepath.name} already exists.")
            return
        logging.info(f"Downloading from {url} ...")
        urllib.request.urlretrieve(url, filepath) 
        logging.info(f"{filepath} downloaded.")
    else:
        logging.warning(f"{filepath} already exists.")

def extract_tarfile(tarfile_path: Path) -> None:
    """
    Extracts contents from any tar.gz file.
    """
    target_path = tarfile_path.parent / tarfile_path.stem
    if not tarfile_path.exists():
        logging.error(f"{tarfile_path} does not exist.")
        return
    if target_path.exists():
        logging.warning(f"{tarfile_path} appears to have been already extracted. Aborting...")
        return
    if not tarfile.is_tarfile(tarfile_path):
        logging.error(f"{tarfile_path} is not a tar archive.")
        return
    with tarfile.open(tarfile_path, "r") as archive:
        archive.extractall(target_path.resolve())
    tarfile_path.unlink()
    logging.info(f"Successfully extracted tar archive to {target_path}.")

def main() -> None:
    urls = DATA_URLS
    data_dir = DATA_DIR

    downloader = functools.partial(download_data, data_dir=data_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        result = executor.map(downloader, urls)

    for path in data_dir.iterdir():
        if not path.suffix == ".gz":
            continue
        extract_tarfile(path)

if __name__ == "__main__":
    main()
