"""
tqdm progress bar from https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
"""

import argparse
from urllib.request import urlretrieve
import zipfile
import os
from tqdm import tqdm
import random
import pandas as pd


def my_hook(t):
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


DOWNLOAD_URL = (
    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-ru.txt.zip"
)

ZIP_FILE_NAME = "en-ru.txt.zip"
ZIP_SRC_NAME = "OpenSubtitles.en-ru.ru"
ZIP_TRG_NAME = "OpenSubtitles.en-ru.en"

SRC_NAME = "ru.txt"
TRG_NAME = "en.txt"
CSV_NAME = "data.csv"


def download(download_path: str) -> None:
    with tqdm(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading"
    ) as t:
        urlretrieve(DOWNLOAD_URL, download_path, reporthook=my_hook(t))


def extract_zip(
    zip_path: str | os.PathLike,
    zip_src_name: str | os.PathLike,
    zip_trg_name: str | os.PathLike,
    data_dir: str | os.PathLike,
    src_path: str | os.PathLike,
    trg_path: str | os.PathLike,
) -> None:
    print("Extracting")
    with zipfile.ZipFile(zip_path) as zip_f:
        zip_f.extract(zip_src_name, path=data_dir)
        zip_f.extract(zip_trg_name, path=data_dir)

    os.rename(os.path.join(data_dir, zip_src_name), src_path)
    os.rename(os.path.join(data_dir, zip_trg_name), trg_path)


def create_csv(
    src_path: str | os.PathLike,
    trg_path: str | os.PathLike,
    csv_path: str | os.PathLike,
    n_rows: int,
):
    print(f"Creating csv ({csv_path}) with size {n_rows}")
    with open(src_path, "r") as src_f, open(trg_path, "r") as trg_f:

        lines = list(zip(src_f.readlines(), trg_f.readlines()))

        lines = random.sample(lines, k=n_rows)

        src_lines = [line[0].strip() for line in lines]
        trg_lines = [line[1].strip() for line in lines]

    pd.DataFrame({"src": src_lines, "trg": trg_lines}).to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./DATA",
        help="Path to the directory where dataset will be downloaded",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200_000,
        help="Number of rows that will be added to resulting dataset. Value -1 represents that all data will be used",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=-1,
        help="Random seed for selecting sample_size rows from dataset. Value -1 represents that seed will not be set",
    )

    args = vars(parser.parse_args())

    data_dir = args["data_dir"]
    src_path = os.path.join(data_dir, SRC_NAME)
    trg_path = os.path.join(data_dir, TRG_NAME)
    csv_path = os.path.join(data_dir, CSV_NAME)

    zip_path = os.path.join(data_dir, ZIP_FILE_NAME)
    download(download_path=zip_path)
    extract_zip(
        zip_path=zip_path,
        zip_src_name=ZIP_SRC_NAME,
        zip_trg_name=ZIP_TRG_NAME,
        data_dir=data_dir,
        src_path=src_path,
        trg_path=trg_path,
    )
    if args["random_seed"] != -1:
        random.seed(args["random_seed"])
    create_csv(
        src_path=src_path,
        trg_path=trg_path,
        csv_path=csv_path,
        n_rows=args["sample_size"],
    )


if __name__ == "__main__":
    main()
