import requests
import zipfile
import io
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split




def download_dataset(url: str, save_path: str):
    """
    Downloads a zip archive from the specified URL and extracts its contents to the specified save path.

    Args:
        url (str): The URL of the zip archive to download.
        save_path (str): The path to save the extracted contents of the zip archive.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request to download the zip archive fails.
        zipfile.BadZipFile: If the downloaded file is not a valid zip archive.
    """

    response = requests.get(url)
    response.raise_for_status()
    
    # Check if the response content is a zip archive
    if zipfile.is_zipfile(io.BytesIO(response.content)):
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            zip_file.extractall(save_path)
        print("File downloaded and unpacked successfully.")
    else:
        print("The downloaded file is not a zip archive.")


def preprocess(df):
    """
    Preprocesses the given DataFrame by swapping the 'reference' and 'translation' columns if 'ref_tox' is less than 'trn_tox',
    dropping rows where 'ref_tox' is less than 'trn_tox', and dropping all columns except 'reference', 'translation', 'ref_tox', and 'trn_tox'.
    
    Args:
    - df: pandas DataFrame containing columns 'reference', 'translation', 'ref_tox', and 'trn_tox'
    
    Returns:
    - pandas DataFrame with preprocessed data
    """
    # if ref_tox < trn_tox then swap columns reference and translation
    df.loc[df['ref_tox'] < df['trn_tox'], ['reference', 'translation']] = df.loc[df['ref_tox'] < df['trn_tox'], ['translation', 'reference']].values

    # drop lines where ref_tox < trn_tox
    df = df[df['ref_tox'] >= df['trn_tox']]

    # drop columns except reference and translation
    df = df.drop(columns=df.columns.difference(['reference', 'translation', 'ref_tox', 'trn_tox']))
    
    return df



def split_dataset(df):
    """
    Split the given DataFrame into train, validation, and test sets.
    Args:
    - df: pandas DataFrame containing columns 'reference', 'translation', 'ref_tox', and 'trn_tox'
    Returns:
    - train, validation, and test sets
    """

    # split df into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # split train_df into train and validation sets
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    return train_df, val_df, test_df


def main(args):
    save_path1 = args.save_path1
    save_path2 = args.save_path2
    save_path3 = args.save_path3

    url1 = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
    download_dataset(url1, save_path1)

    df = pd.read_csv(save_path1 + "filtered.tsv", sep="\t", index_col=0)
    df = preprocess(df)

    # save preprocesed df to save_path3 + "filtered.csv"
    df.to_csv(save_path3 + "filtered.csv", index=False)

    #save df['reference'] as txt file
    df['reference'].to_csv(save_path3 + "reference.txt", index=False, header=False)
    df['translation'].to_csv(save_path3 + "translation.txt", index=False, header=False)


    url2 = "https://www.kaggle.com/datasets/nicapotato/bad-bad-words/download?datasetVersionNumber=1"
    download_dataset(url2, save_path2)
    txt = pd.read_csv(save_path2 + "bad-words.csv")
    txt.to_csv(save_path3 + "bad_words.txt", index=False, header=False)


    # split df into train, validation, and test sets
    train_df, val_df, test_df = split_dataset(df)

    # save train, validation, and test sets to save_path3
    train_df.to_csv(save_path3 + "train.csv", index=False)
    val_df.to_csv(save_path3 + "validation.csv", index=False)
    test_df.to_csv(save_path3 + "test.csv", index=False)

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path1", help="save path to raw dataset", default="data/raw/")
    parser.add_argument("--save_path2", help="save path to external datasets", default='data/external/')
    parser.add_argument("--save_path3", help="save path to interim", default='data/interim/')

    args = parser.parse_args()


    main(args)
