"""
Short script to download TCRdb sequences. These are automatically saved as gzipped tsv files.
"""
import hashlib
import gzip
import logging
import argparse
from typing import Optional
import shutil
import urllib.request

import tqdm

logging.basicConfig(level=logging.INFO)


def md5_file(fname: str) -> str:
    """Return md5sum of file contents"""
    # https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
    hasher = hashlib.md5()
    with open(fname, "rb") as source:
        for chunk in iter(lambda: source.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_accession(accession: str, fname: Optional[str] = None) -> str:
    """Download the given accession"""
    url = f"http://bioinfo.life.hust.edu.cn/TCRdb/Download/{accession}.tsv"
    dest = fname if fname is not None else f"{accession}.tsv.gz"
    with urllib.request.urlopen(url) as response:
        with gzip.open(dest, "wb") as sink:
            shutil.copyfileobj(response, sink)
    return dest


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "accessions", type=str, help="txt file with accessions, newline delimited"
    )
    return parser


def main():
    """Run script"""
    args = build_parser().parse_args()

    # Read accession list
    with open(args.accessions) as source:
        accessions = [l.strip() for l in source if not l.startswith("#")]
    logging.info(f"Read in {len(accessions)} accessions")

    # Download each
    for acc in accessions:
        fname = download_accession(acc)
        logging.info(f"{acc}\t{fname}\t{md5_file(fname)}")


if __name__ == "__main__":
    main()
