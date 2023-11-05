import argparse
import pandas as pd
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import warnings 
warnings.filterwarnings('ignore')


def load_model(model_path):

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    model.eval()

    model.config.use_cache = False
    return model

def detoxificate(model, tokenizer, text):
    prefix="paraphrase:"
    input = prefix + text
    input_ids = tokenizer(input, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids,  num_beams=4, )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    

def predict(model, tokenizer, test_df_path):


    with open(test_df_path, 'r') as f:
        test_df = f.readlines()

    result = []
    for text in tqdm(test_df):
        result.append(detoxificate(model, tokenizer, text))

    # save result to txt file   
    with open("../data/interim/result.txt", "w") as f:
        for text in result:
            f.write(text + "\n")

    return result   


def main(args):
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = load_model(args.model_path)
    result = predict(model, tokenizer, args.data_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to model", default="../models/t5-small/t5-small-tuned")
    parser.add_argument("--data_path", help="path test data", default="../data/interim/test_reference.txt")
    args = parser.parse_args()


    main(args)