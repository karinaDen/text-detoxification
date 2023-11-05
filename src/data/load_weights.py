import argparse
import gdown
import zipfile



def download_and_extract(url: str, extract_dir: str):
    """
    Downloads a zip archive from the specified URL and extracts its contents to the specified save path.

    Args:
        url (str): The URL of the zip archive to download.
        extract_dir (str): The path to save the extracted contents of the zip archive.  
    """

    output = extract_dir + "best-model.zip"
    gdown.download(url, extract_dir, quiet=False, fuzzy=True) 

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)




def main(args):
    URL = 'https://drive.google.com/file/d/1TS2FeNofbu_bydF3AaxcszcWs6htSk-G/view?usp=sharing'
    download_and_extract(URL, args.save_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", help="path to where to load weights", default="../models/t5-small/")
    args = parser.parse_args()


    main(args)