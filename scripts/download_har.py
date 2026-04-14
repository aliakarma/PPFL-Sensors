import argparse
import os
import urllib.request
import zipfile

def download_file(url, out_path):
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, out_path)
    print(f"Downloaded to {out_path}.")

def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Download the UCI HAR dataset.")
    parser.add_argument("--out_dir", type=str, default="data/raw/", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    zip_path = os.path.join(args.out_dir, "dataset.zip")
    har_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

    download_file(har_url, zip_path)
    unzip_file(zip_path, args.out_dir)

    print("UCI HAR Dataset successfully downloaded and extracted.")

if __name__ == "__main__":
    main()