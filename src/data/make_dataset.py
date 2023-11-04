import requests
import zipfile
import io
import pandas as pd
import argparse



def download_dataset(url: str, save_path: str):
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
    # if ref_tox < trn_tox then swap columns reference and translation
    df.loc[df['ref_tox'] < df['trn_tox'], ['reference', 'translation']] = df.loc[df['ref_tox'] < df['trn_tox'], ['translation', 'reference']].values

    # drop lines where ref_tox < trn_tox
    df = df[df['ref_tox'] >= df['trn_tox']]

    # drop columns except reference and translation
    df = df.drop(columns=df.columns.difference(['reference', 'translation', 'ref_tox', 'trn_tox']))
    
    return df





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

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path1", help="save path to filtered_paranmt.zip", default="data/raw/")
    parser.add_argument("--save_path2", help="save path to bad-words.csv", default='data/external/')
    parser.add_argument("--save_path3", help="save path to interim", default='data/interim/')

    args = parser.parse_args()


    main(args)
